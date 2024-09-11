from datasets import VaccinChat


def test_chatbot():
    vaccinchat = VaccinChat(inputs="fbank", outputs="labels")
    dataset = vaccinchat("train")
    print(f"{vaccinchat.attributes['name']} M={len(dataset)} Nx={dataset.input_dim} Ny={dataset.output_dim}")
    X, l, y = dataset[:32]
    print(f"Input: {X.size()} Output: {y.size()} L={l.max()}")


if __name__ == "__main__":
    test_chatbot()
