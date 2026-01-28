from trainkeeper.experiment import run_reproducible


@run_reproducible()
def demo():
    print("TrainKeeper works.")


if __name__ == "__main__":
    demo()
