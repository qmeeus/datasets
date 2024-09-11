

def test_grabo():
    from datasets import Grabo
    grabo = Grabo()
    pp10 = grabo("pp10")
    print(len(pp10))
    print(pp10.dims)


if __name__ == "__main__":
    test_grabo()