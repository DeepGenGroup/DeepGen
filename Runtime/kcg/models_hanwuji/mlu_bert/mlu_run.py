import torch
import torch_mlu
from mlu_model import ModelArgs, BERT

def test_model(model, input):
    times = []
    for i in range(20):
        start_event = torch.mlu.Event(enable_timing=True)
        end_event = torch.mlu.Event(enable_timing=True)
        start_event.record()

        output = model(input)

        end_event.record()
        torch.mlu.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        times.append(elapsed_time)
    times = sorted(times)
    mid_time = times[10]
    return mid_time


def run(seq_len=1024, vocab_size=32000, base="our"):
    model = BERT(base)
    batchs = [1, 2, 4]
    print("seq_len: {}".format(seq_len))
    for batch in batchs:
        input = torch.randint(low=1, high=vocab_size, size=(batch, seq_len), dtype=torch.long).to("mlu")
        model.to("mlu")
        cost = test_model(model, input)
        print("batch: {} time cost: {}".format(batch, cost))

if __name__ == "__main__":
    # run(base="baseline")
    run()
