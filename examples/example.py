import numpy as np
import sys; sys.path.insert(0, "build") 
import runlab_py as rl


def main():
    engine = rl.Engine()
    engine.add_node("input", "input", {"data": np.array([1.0, 2.0, 3.0, 4.0])})
    engine.add_node("scaled", "scale", {"input": "input", "factor": 1.5})
    engine.add_node("embed", "embedding", {"input": "scaled"})
    engine.add_node("total", "sum", {"input": "embed"})

    engine.run()

    print("embed:", engine.get_vector("embed"))
    print("sum:", engine.get_float("total"))


if __name__ == "__main__":
    main()
