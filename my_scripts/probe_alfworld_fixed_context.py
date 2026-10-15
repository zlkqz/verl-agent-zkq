#!/usr/bin/env python3
"""Compare checkpoints on identical healthy continuations, without training.

Uses one CUDA device. FP32 entropy/NLL at T=1, 0.8 and 0.4 distinguish changes
on fixed contexts from changes in on-policy rollout contexts. This does not
test vLLM/FSDP alignment or measure interactive task success.
"""
import argparse
import gc
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', required=True, type=Path)
    parser.add_argument('--model', required=True, action='append', help='HF model directory; repeat for each checkpoint')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--limit', type=int, default=12)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError('This diagnostic needs one CUDA device; no training is performed.')
    samples = [json.loads(line) for line in args.samples.read_text().splitlines() if line.strip()][:args.limit]
    if not samples:
        raise ValueError('No fixed contexts supplied')
    results = {'samples': str(args.samples.resolve()), 'models': []}
    temperatures = [1.0, 0.8, 0.4]
    for model_path in args.model:
        print(f'Loading {model_path}', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map={'': args.device},
        ).eval()
        record = {'model': str(Path(model_path).resolve()), 'rows': []}
        totals = {str(t): {'tokens': 0, 'entropy_sum': 0., 'nll_sum': 0., 'top1_count': 0} for t in temperatures}
        with torch.inference_mode():
            for sample in samples:
                prompt = tokenizer.encode(sample['prompt'], add_special_tokens=False)
                response = tokenizer.encode(sample['response'], add_special_tokens=False)
                if not prompt or not response:
                    raise ValueError(f'Empty prompt/response: {sample["id"]}')
                ids = torch.tensor([prompt + response], device=args.device)
                output = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
                # Predict each original healthy response token, with the same
                # original prefix at every checkpoint (teacher forcing).
                logits = output.logits[0, len(prompt) - 1:-1]
                labels = ids[0, len(prompt):]
                row = {'id': sample['id'], 'response_tokens': len(response), 'temperatures': {}}
                for temperature in temperatures:
                    entropy_sum = nll_sum = 0.
                    initial_entropy = eos_nll = None
                    top1 = 0
                    # Bound temporary FP32 vocabulary tensors on an 8B model.
                    for start in range(0, len(response), 64):
                        logp = (logits[start:start + 64].float() / temperature).log_softmax(-1)
                        target = labels[start:start + 64]
                        entropies = -(logp.exp() * logp).sum(-1)
                        entropy_sum += float(entropies.sum().item())
                        nll_sum += float(-logp.gather(-1, target[:, None]).sum().item())
                        top1 += int((logp.argmax(-1) == target).sum().item())
                        if start == 0:
                            initial_entropy = float(entropies[0].item())
                        if start + 64 >= len(response) and response[-1] == tokenizer.eos_token_id:
                            eos_nll = float(-logp[-1, response[-1]].item())
                        del logp, entropies
                    count = len(response)
                    row['temperatures'][str(temperature)] = {
                        'entropy': entropy_sum / count, 'nll': nll_sum / count,
                        'top1_accuracy': top1 / count,
                        'initial_entropy': initial_entropy, 'eos_nll': eos_nll,
                    }
                    total = totals[str(temperature)]
                    total['tokens'] += count
                    total['entropy_sum'] += entropy_sum
                    total['nll_sum'] += nll_sum
                    total['top1_count'] += top1
                record['rows'].append(row)
                del output, logits, labels, ids
        record['summary'] = {
            temperature: {'tokens': value['tokens'],
                          'entropy': value['entropy_sum'] / value['tokens'],
                          'nll': value['nll_sum'] / value['tokens'],
                          'top1_accuracy': value['top1_count'] / value['tokens']}
            for temperature, value in totals.items()
        }
        for temperature in totals:
            rows = [row['temperatures'][temperature] for row in record['rows']]
            record['summary'][temperature]['initial_entropy_mean'] = sum(row['initial_entropy'] for row in rows) / len(rows)
            eos_rows = [row['eos_nll'] for row in rows if row['eos_nll'] is not None]
            record['summary'][temperature]['eos_nll_mean'] = sum(eos_rows) / len(eos_rows) if eos_rows else None
        results['models'].append(record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))
        print(json.dumps({'model': model_path, 'summary': record['summary']}), flush=True)
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
