"""LFMA experiment preparation, evaluation, and prediction."""

import torch
import torch.nn.functional as F
import os
import argparse
import json
from collections.abc import Mapping
from contextlib import nullcontext
from lfma.data.streams import task_metrics, make_loader, TextCollator
from lfma.core import (
    AverageMeter,
    GLUE_TASKS,
    validate_config,
    insert_adapters,
    load_backbone,
    cuda_device,
    load_config,
    set_seed,
    BACKBONES,
    IMAGE_TASKS,
    IMAGE_HUB,
)
from lfma.adaptation.injection import merge_adapters, inject_adapters
from pathlib import Path
from lfma.artifacts.storage import (
    load_pretrained_adapter,
    read_metadata,
    save_pretrained_adapter,
    restore_training_state,
    write_merge_manifest,
)
from PIL import Image
from torch import nn
from lfma.experiments.protocols import (
    inspect_adapter_cli,
    summarize_cli,
    grid_cli,
    run_protocol_cli,
    catalog_cli,
)


def configure_optimizer(model, cfg):
    head_ids = (
        {id(p) for p in model.classifier.parameters()}
        if hasattr(model, 'classifier')
        else set()
    )
    adapter = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    head = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    groups = [{'params': adapter, 'lr': cfg['learning_rate']}]
    if head:
        groups.append({'params': head, 'lr': cfg['head_learning_rate']})
    return torch.optim.AdamW(groups, weight_decay=cfg['weight_decay'])


def run_epoch(model, loader, device, optimizer=None, config=None):
    config = config or {
        'data': {'task': 'classification'},
        'train': {'gradient_accumulation': 1, 'precision': 'fp32'},
    }
    training, task = optimizer is not None, config['data']['task']
    schedule = config['train']
    accumulation = schedule['gradient_accumulation']
    model.train(training)
    meter, predictions, references = AverageMeter(), [], []
    window_examples = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for step, item in enumerate(loader):
        batch = (
            {k: v.to(device, non_blocking=True) for k, v in item.items()}
            if isinstance(item, Mapping)
            else None
        )
        labels = batch.get('labels') if batch is not None else item[1].to(device)
        amp = (
            torch.autocast('cuda', dtype=torch.bfloat16)
            if schedule['precision'] == 'bf16'
            else nullcontext()
        )
        with torch.set_grad_enabled(training), amp:
            if batch is None:
                logits = model(item[0].to(device))
                loss = F.cross_entropy(logits, labels)
            else:
                output = model(**batch)
                logits, loss = output.logits, output.loss
            if training:
                if loss is None:
                    raise ValueError('Training examples must contain supervised labels')
                (loss * len(labels)).backward()
                window_examples += len(labels)
        if training and ((step + 1) % accumulation == 0 or step + 1 == len(loader)):
            parameters = [p for p in model.parameters() if p.requires_grad]
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.div_(window_examples)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            window_examples = 0
        estimate = logits.detach().flatten() if task == 'stsb' else logits.detach().argmax(-1)
        predictions.extend(
            estimate.float().cpu().tolist() if task == 'stsb' else estimate.cpu().tolist()
        )
        if labels is not None:
            references.extend(labels.detach().cpu().tolist())
            meter.update(loss.detach().float().item(), len(labels))
    result = {'samples': len(predictions)}
    if references:
        result.update(loss=meter.avg, **task_metrics(task, predictions, references))
    return result, predictions


def apply_runtime_options(config, args):
    for argument, section, key in [
        ('cache_dir', 'model', 'cache_dir'),
        ('model_dir', 'model', 'local_dir'),
        ('data_root', 'data', 'root'),
        ('dataset_dir', 'data', 'local_dir'),
        ('imagefolder', 'data', 'imagefolder'),
        ('manifest_dir', 'data', 'manifest_dir'),
    ]:
        value = getattr(args, argument, None)
        if value is not None:
            config[section][key] = value
    if getattr(args, 'offline', False):
        config['model']['offline'] = True
    if config['model']['offline']:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_DATASETS_OFFLINE'] = '1'
    return config


#!/usr/bin/env python3


def train_main(args):
    device = cuda_device(args.device)
    if args.resume and args.config is None:
        config = read_metadata(args.resume)['config']
        from lfma.core import apply_overrides

        config = apply_overrides(config, args.opts)
    else:
        config = load_config(args.config or 'config.yaml', args.opts)
    config = apply_runtime_options(config, args)
    validate_config(config)
    set_seed(config['train']['seed'])
    torch.set_num_threads(config['train']['num_threads'])
    output = Path(config['train']['save_dir'])
    output.mkdir(parents=True, exist_ok=True)
    epoch, best = 0, float('-inf')
    if args.resume:
        saved = read_metadata(args.resume)
        for section in ('adapter',):
            if saved['config'][section] != config[section]:
                raise ValueError('Resume adapter configuration must match the checkpoint')
        if (
            saved['config']['model']['backbone'] != config['model']['backbone']
            or saved['config']['data']['task'] != config['data']['task']
        ):
            raise ValueError('Resume backbone and task must match the checkpoint')
        config['model']['revision'] = saved['config']['model']['revision']
        model, processor, config, metadata = load_pretrained_adapter(
            args.resume, device, config
        )
        epoch, best = metadata['epoch'], metadata['best_score']
    else:
        model, processor = load_backbone(config, device)
        insert_adapters(model, config)
    if config['train']['epochs'] <= epoch:
        raise ValueError('Training epochs must exceed the resumed checkpoint epoch')
    optimizer = configure_optimizer(model, config['train'])
    if args.resume:
        restore_training_state(args.resume, optimizer, config['train'], device)
    train_loader, val_loader = (
        make_loader(config, processor, split) for split in ('train', 'validation')
    )
    primary = GLUE_TASKS.get(config['data']['task'], (None, None, 'accuracy'))[2]
    history_path = output / 'history.jsonl'
    with history_path.open('a' if args.resume else 'w') as history:
        for current in range(epoch, config['train']['epochs']):
            if hasattr(train_loader.sampler, 'set_epoch'):
                train_loader.sampler.set_epoch(current)
            train_scores, _ = run_epoch(model, train_loader, device, optimizer, config)
            scores, _ = run_epoch(model, val_loader, device, config=config)
            improved = scores[primary] > best
            best = max(best, scores[primary])
            record = {'epoch': current + 1, 'train': train_scores, 'validation': scores}
            history.write(json.dumps(record) + '\n')
            history.flush()
            save_pretrained_adapter(
                model, processor, config, output / 'last', optimizer, current + 1, best
            )
            if improved:
                save_pretrained_adapter(
                    model, processor, config, output / 'best', optimizer, current + 1, best
                )
            print(record, flush=True)
    counts = {
        'adapter_real_scalars': sum(
            p.numel() for n, p in model.named_parameters() if n.endswith('.c')
        ),
        'head_parameters': sum(p.numel() for p in model.classifier.parameters()),
        'best_validation_metric': primary,
        'best_validation_score': best,
        'epoch': config['train']['epochs'],
    }
    (output / 'metrics.json').write_text(json.dumps(counts, indent=2) + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--opts', nargs='*', default=[])
    parser.add_argument('--resume', help='LFMA adapter checkpoint directory')
    parser.add_argument('--cache-dir')
    parser.add_argument('--model-dir', help='Local pretrained encoder directory')
    parser.add_argument('--data-root')
    parser.add_argument(
        '--dataset-dir', help='Hugging Face DatasetDict saved with save_to_disk'
    )
    parser.add_argument('--imagefolder', help='Local train/validation/test class folders')
    parser.add_argument('--manifest-dir', help='Portable JSONL benchmark bundle')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def train_cli():
    train_main(parse_args())


#!/usr/bin/env python3


def eval_main(args):
    device = cuda_device(args.device)
    config = apply_runtime_options(read_metadata(args.checkpoint)['config'], args)
    model, processor, config, _ = load_pretrained_adapter(args.checkpoint, device, config)
    loader = make_loader(config, processor, args.split)
    scores, predictions = run_epoch(model, loader, device, config=config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'metrics.json').write_text(json.dumps(scores, indent=2) + '\n')
    (output / 'predictions.tsv').write_text(
        'index\tprediction\n'
        + ''.join(f'{i}\t{value}\n' for i, value in enumerate(predictions))
    )
    if args.merge:
        merged = merge_adapters(model)
        merged.save_pretrained(args.merge, safe_serialization=True)
        processor.save_pretrained(args.merge)
        write_merge_manifest(args.merge, config, args.checkpoint)
    print(scores)


def eval_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--cache-dir')
    parser.add_argument('--model-dir')
    parser.add_argument('--data-root')
    parser.add_argument('--dataset-dir')
    parser.add_argument('--imagefolder')
    parser.add_argument('--manifest-dir')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='results/evaluation')
    parser.add_argument(
        '--merge', help='Save a standard merged Transformers model in this directory'
    )
    eval_main(parser.parse_args())


def predict_main(args):
    device = cuda_device(args.device)
    config = apply_runtime_options(read_metadata(args.checkpoint)['config'], args)
    model, processor, config, _ = load_pretrained_adapter(args.checkpoint, device, config)
    model.eval()
    family = BACKBONES[config['model']['backbone']]['family']
    if family == 'vit':
        if not args.images:
            raise ValueError('Image models require --images')
        records = args.images
    else:
        if not args.text:
            raise ValueError('Text models require --text JSONL using task text fields')
        records = [
            json.loads(line)
            for line in Path(args.text).read_text().splitlines()
            if line.strip()
        ]
    predictions = []
    with torch.no_grad():
        for start in range(0, len(records), args.batch_size):
            rows = records[start : start + args.batch_size]
            if family == 'vit':
                images = []
                for path in rows:
                    with Image.open(path) as image:
                        images.append(image.convert('RGB'))
                batch = processor(images=images, return_tensors='pt')
            else:
                batch = TextCollator(processor, config)(rows)
                batch.pop('labels', None)
            logits = model(**{k: v.to(device) for k, v in batch.items()}).logits
            values = (
                logits.flatten().tolist()
                if config['data']['task'] == 'stsb'
                else logits.argmax(-1).tolist()
            )
            predictions.extend(values)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'predictions': predictions}, indent=2) + '\n')


def predict_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--images', nargs='+')
    parser.add_argument('--text')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--model-dir')
    parser.add_argument('--cache-dir')
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default='results/predictions.json')
    arguments = parser.parse_args()
    if arguments.batch_size < 1:
        parser.error('--batch-size must be positive')
    predict_main(arguments)


def prepare_main(args):
    if args.backbone:
        from huggingface_hub import snapshot_download

        model = BACKBONES[args.backbone]
        path = snapshot_download(
            model['id'],
            revision=model['revision'],
            cache_dir=args.cache_dir,
            local_dir=args.local_dir,
            allow_patterns=[
                '*.safetensors',
                'config.json',
                'preprocessor_config.json',
                'tokenizer*.json',
                'vocab.json',
                'merges.txt',
                'special_tokens_map.json',
            ],
        )
        print(f'Pretrained {args.backbone}: {path}')
    if args.task:
        config = load_config('config.yaml')
        config['data'].update(task=args.task, root=args.data_root, cache_dir=args.dataset_cache)
        if args.task in GLUE_TASKS:
            from lfma.data.streams import dataset_from_hub
            from lfma.core import GLUE_REVISION

            data = dataset_from_hub(config, 'nyu-mll/glue', args.task, GLUE_REVISION)
            if args.dataset_dir:
                data.save_to_disk(args.dataset_dir)
            print({split: len(records) for split, records in data.items()})
        elif args.task in IMAGE_HUB and args.dataset_dir:
            from datasets import DatasetDict
            from lfma.data.streams import dataset_from_hub

            repo, revision = IMAGE_HUB[args.task]
            splits = (
                ('train', 'validation', 'test')
                if args.task == 'resisc45'
                else ('train', 'test')
            )
            data = DatasetDict(
                {
                    split: dataset_from_hub(
                        config,
                        repo,
                        revision=revision,
                        split=split,
                        data_files={split: f'data/{split}-*.parquet'},
                    )
                    for split in splits
                }
            )
            data.save_to_disk(args.dataset_dir)
            print({split: len(records) for split, records in data.items()})
        else:
            from lfma.data.streams import read_vision

            for split in ('train', 'validation', 'test'):
                dataset = read_vision(config, split)
                print(f'{args.task}/{split}: {len(dataset)} images')
    if not args.backbone and not args.task:
        raise ValueError('Select --backbone and/or --task')


def prepare_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', choices=list(BACKBONES))
    parser.add_argument('--task', choices=list(GLUE_TASKS) + list(IMAGE_TASKS))
    parser.add_argument('--cache-dir', default='.cache/huggingface')
    parser.add_argument('--local-dir', help='Portable destination for pretrained model files')
    parser.add_argument('--data-root', default='datasets')
    parser.add_argument('--dataset-cache', default='.cache/datasets')
    parser.add_argument(
        '--dataset-dir', help='Save a portable GLUE/Cars/RESISC45 DatasetDict here'
    )
    prepare_main(parser.parse_args())


def frozen_projection_main(args):
    args.device = cuda_device(args.device)
    torch.manual_seed(42)
    encoder = nn.Sequential(nn.Linear(12, 8), nn.Tanh(), nn.Linear(8, 4)).to(args.device)
    inject_adapters(encoder, ['0', '2'], top_k_ratio=0.25)
    optimizer = torch.optim.AdamW((p for p in encoder.parameters() if p.requires_grad), lr=0.05)
    x, target = torch.randn(32, 12, device=args.device), torch.randn(32, 4, device=args.device)
    for _ in range(5):
        optimizer.zero_grad(set_to_none=True)
        loss = (encoder(x) - target).square().mean()
        loss.backward()
        optimizer.step()
    torch.testing.assert_close(encoder(x), merge_adapters(encoder)(x))
    print(f'projection loss: {loss.item():.4f}')


def projection_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    frozen_projection_main(parser.parse_args())


def main():
    import sys

    from lfma.artifacts.validation import adapter_cli
    from lfma.spectrum import spectrum_cli
    from lfma.data.manifest import manifest_cli
    from lfma.experiments.session import plan_cli
    from lfma.artifacts.bundle import bundle_cli
    from lfma.experiments.comparison import compare_cli
    from lfma.experiments.fit import fit_cli, evaluate_detailed_cli, diagnostics_cli, compose_cli
    from lfma.experiments.analysis import analyze_cli, calibrate_cli
    from lfma.experiments.prepare import data_audit_cli, subset_bundle_cli, budget_cli, budget_grid_cli

    commands = {
        'fit': fit_cli,
        'evaluate-detailed': evaluate_detailed_cli,
        'analyze': analyze_cli,
        'calibrate': calibrate_cli,
        'audit-data': data_audit_cli,
        'subset': subset_bundle_cli,
        'budget': budget_cli,
        'budget-grid': budget_grid_cli,
        'diagnostics': diagnostics_cli,
        'compose': compose_cli,
        'adapter': adapter_cli,
        'spectrum': spectrum_cli,
        'manifest': manifest_cli,
        'plan': plan_cli,
        'bundle': bundle_cli,
        'compare': compare_cli,
        'train': train_cli,
        'evaluate': eval_cli,
        'predict': predict_cli,
        'prepare': prepare_cli,
        'inspect': inspect_adapter_cli,
        'summarize': summarize_cli,
        'grid': grid_cli,
        'protocol': run_protocol_cli,
        'catalog': catalog_cli,
        'projection': projection_cli,
    }
    parser = argparse.ArgumentParser(
        description='Layerwise Fourier masked adapter experiments.'
    )
    parser.add_argument('task', choices=tuple(commands))
    task = parser.parse_args(sys.argv[1:2]).task
    sys.argv = [sys.argv[0] + ' ' + task, *sys.argv[2:]]
    commands[task]()


if __name__ == "__main__":
    main()
