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
    plt.savefig(f"results/plots/loss_graph_{fig_name}.jpeg", dpi=150, bbox_inches='tight')
    plt.show()


def visualize_performance(y_train, y_hat, fig_name):
    plt.figure(figsize = (10,6), dpi =150)
    plt.scatter(y_train, y_hat, color = 'steelblue',alpha = 0.6, s=8)
    lims = [min(y_train.min(), y_hat.min()),
            max(y_train.max(),y_hat.max())]
    plt.plot(lims,lims, 'r--', linewidth =2, label = "Prediction" )
    plt.xlabel("true value")
    plt.ylabel("predicted value")
    plt.grid(True, alpha = 0.5)
    plt.tight_layout()
    plt.savefig(f"results/plots/evaluation_{fig_name}.jpeg", dpi =150, bbox_inches = 'tight')
    plt.show()

    residuals = y_hat - y_train
    plt.figure(figsize=(10,6), dpi = 150)
    plt.hist(residuals,bins =50, color = 'coral', alpha = 0.6)
    plt.axvline(0, color='black', linewidth=1.5, label='Zero residual')
    plt.axvline(residuals.mean(), color='red', linestyle='--',
                label=f'Mean residual = {residuals.mean():.3f}')
    plt.xlabel('prediction_error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha = 0.5)
    plt.tight_layout()
    plt.savefig(f"results/plots/histogram_{fig_name}.jpeg")
    plt.show()


def save_model(model, path:str):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def load_model(model, path:str):
    model.load_state_dict(torch.load(path))
    return model
