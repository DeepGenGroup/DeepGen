from mlu_model import LLAMA2
import torch
import torch_mlu
import torch.nn.functional as F


def test_model(model, input):
    times = []
    repeat = 10
    for i in range(repeat):
        start_event = torch.mlu.Event(enable_timing=True)
        end_event = torch.mlu.Event(enable_timing=True)
        start_event.record()

        output = model(input)

        end_event.record()
        torch.mlu.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        times.append(elapsed_time)
    times = sorted(times)
    mid_time = times[int(repeat/2)]
    return mid_time
    

def run(seq_len=512, vocab_size=32000, base="our"):
    model = LLAMA2(base)
    batchs = [1]
    print("seq_len: {}".format(seq_len))
    for batch in batchs:
        input = torch.randint(low=1, high=vocab_size, size=(batch, seq_len), dtype=torch.long).to("mlu")
        model.to("mlu")
        cost = test_model(model, input)
        print("batch: {} time cost: {}".format(batch, cost))


if __name__ == "__main__":
    run(base="baseline")
    # run()