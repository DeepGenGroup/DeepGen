from model import GPT2
import torch, json
import torch.nn.functional as F


def test_model(model, input):
    times = []
    repeat = 10
    for i in range(repeat):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

        output = model(input)

        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        times.append(elapsed_time)
    times = sorted(times)
    mid_time = times[int(repeat/2)]
    return mid_time


def run(max_seq_len=1024, max_batch_size=32, vocab_size=32000):
    device = torch.device("cuda")
    args = ModelArgs()
    model = GPT2()
    batchs = [1, 2, 4, 8]
    print("seq_len: {}".format(max_seq_len))
    for batch in batchs:
        input = torch.randint(low=1, high=vocab_size, size=(batch, max_seq_len), dtype=torch.long, device=device)
        model = model.to(device)
        cost = test_model(model, input)
        print("batch: {} time cost: {}".format(batch, cost))


if __name__ == "__main__":
  run()
