import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import torchvision


def main() -> None:
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    trainset = datasets.MNIST(
        "mnist_data/", download=True, train=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=1000, shuffle=True, num_workers=0
    )

    def show_image(img, label):
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
        plt.title(f"Label: {label}")
        plt.show()

    dataiter = iter(trainloader)
    images, labels = next(dataiter)

    # show_image(images[0].torch.squeeze(), labels[0].item())

    # plt.imshow(torchvision.utils.make_grid(images))
    #
    # print(" ".join(f"{labels[j]}" for j in range(4)))

    examples = enumerate(trainloader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)

    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
