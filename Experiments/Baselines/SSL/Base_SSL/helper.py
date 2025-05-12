from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn.functional as F

class KerasLikeTorchTrainer:
    def __init__(self, model, loss_fn, optimizer, device=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def fit(self, x_train, y_train, epochs=1, batch_size=32, validation_data=None):
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch in train_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

            acc = correct / total
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}, acc = {acc:.4f}")

            if validation_data:
                self.evaluate(*validation_data)

    def evaluate(self, x_val, y_val, batch_size=32):
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size)
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        acc = correct / total
        avg_loss = total_loss / len(val_loader)
        print(f"Validation: loss = {avg_loss:.4f}, acc = {acc:.4f}")
        return avg_loss, acc

    def predict(self, x, batch_size=32):
        loader = DataLoader(x, batch_size=batch_size)
        self.model.eval()
        all_outputs = []

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                outputs = self.model(batch)
                probs = F.softmax(outputs, dim=1)
                all_outputs.append(probs.cpu())

        return torch.cat(all_outputs)
