import torch

def train(model, optimizer, scheduler, train_dataloader, device, loss_fn, epochs):
    for epoch_i in range(epochs):
        total_loss = 0

        model.train()

        for i, batch in enumerate(train_dataloader):

            print(f"Batch: {i}")

            batch_ids, batch_attention_mask, batch_label = \
                batch['ids'], batch['attention_mask'], batch['label']

            batch_ids, batch_attention_mask, batch_label = \
                batch_ids.to(device, dtype=torch.long), batch_attention_mask.to(device, dtype=torch.long), batch_label.to(device, dtype=torch.long)

            model.zero_grad()

            outputs = model(batch_ids, batch_attention_mask)

            loss = loss_fn(outputs, batch_label)
            total_loss += loss.item()

            loss.backward()

            optimizer.step()
            scheduler.step()


        avg_train_loss = total_loss / len(train_dataloader)

        print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f}")
