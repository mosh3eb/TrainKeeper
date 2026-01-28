from trainkeeper.experiment import run_reproducible


@run_reproducible(auto_capture_git=True)
def train():
    print("hello, trainkeeper")


if __name__ == "__main__":
    train()
