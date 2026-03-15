import matplotlib.pyplot as plt
import torch


def plot_losses(train_loss:list, val_loss:list, fig_name:str):
    plt.figure(figsize = (10,6), dpi = 150)
    plt.plot(train_loss, linewidth = 2, color = "royalblue" ,label = "Training Loss")
    plt.plot(val_loss, linewidth = 2, linestyle = '--', color = "teal", label = "Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha = 0.5)
    plt.tight_layout()
    plt.savefig(f"../results/plots/loss_graph_{fig_name}.png ", dpi=150, bbox_inches='tight')
    plt.show()


def visualize_performance(y_train, y_hat, fig_name):
    pass


def save_model(model, path:str):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path:str):
    model.load_state_dict(torch.load(path))
    return model
