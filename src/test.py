from datasets import load_dataset
from utils import process_DiffusionDreamDataset
from torch.utils.data import DataLoader
import torch
from data import collate_DiffusiosnDream

print("bb")
dataset = load_dataset('fffffchopin/DiffusionDream_Dataset', streaming=True,split='train')
#dataset = dataset.train_test_split(test_size=0.2)
print(dataset)


def collate_fn(examples):
    print(examples[0]['low_res'][0].shape)
    print(examples[0]['full_res'][0].shape)
    print(examples[0]['action'][0].shape)
    print(examples[0]['low_res'].__class__)
    print(len(examples),examples.__class__)
    original_pixel_values = torch.stack(
        [example["low_res"] for example in examples])
    original_pixel_values = original_pixel_values.to(
        memory_format=torch.contiguous_format).float()
    edited_pixel_values = torch.stack(
        [example["full_res"] for example in examples])
    edited_pixel_values = edited_pixel_values.to(
        memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["actions"] for example in examples])
    return {
        "original_pixel_values": original_pixel_values,
        "edited_pixel_values": edited_pixel_values,
        "input_ids": input_ids,
    }

def map_fn(x):
    print(x['previous_frame_1'].shape)
    return x
print("aa")
dataset = dataset.map(process_DiffusionDreamDataset)
dataset = dataset.batch(batch_size=9)
print("dd")
dataloader = DataLoader(dataset, batch_size=16,collate_fn=collate_DiffusiosnDream)
print(":ff")
#print(len(dataloader))
for i, batch in enumerate(dataloader):
    print(batch.obs.shape)
    print(batch.act.shape)
    print(batch.rew.shape)
    print(batch.info.__class__)
    print(batch.info[0]['full_res'].shape)
    print(batch.segment_ids.shape)
    print(batch.mask_padding)
    break
#dataset = dataset.batch(batch_size=4)
#dataset = iter(dataset)
#print(next(dataset))

