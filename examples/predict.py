"""Direct context inference, including possible infinity and undefined results."""
import argparse
from teenie import EpisodeSampler, load_model
from teenie.data import load_tasks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', default='checkpoints/residual.best.pt')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--functions', default='examples/composite.py')
    args = parser.parse_args()
    model, _ = load_model(args.load, args.device)
    episode = EpisodeSampler(load_tasks(args.functions), seed=42).sample(1)
    prediction = model.predict(episode.context[0].tolist(), episode.query[0].tolist())
    print('query:', episode.query[0].tolist(), 'prediction:', prediction, 'expected:', episode.target.item())


if __name__ == '__main__':
    main()
