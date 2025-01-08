"""Entry point of every pipeline. Actually run pipeline with use
specified in the congig pipeline runner. Config is a yaml file processed
by Hydra which is stored in the configs directory"""

import logging
from collections import namedtuple
from pathlib import Path
from typing import Any

import hydra
import mlflow
import networkx as nx
from omegaconf import DictConfig, OmegaConf

from src.pipelines.runners import PipelineRunnerBase
from src.pipelines.transforms_base import CachedTransform
from src.utils.hydra.instantiate import instantiate_recursive

Transform = namedtuple("Transform", ["name", "deps", "transform"])

logger = logging.getLogger(__file__)


def instantiate_transforms(transforms_cfg: DictConfig) -> list[Transform]:
    result = []
    for name, transform in transforms_cfg.items():
        transform["transform"]._set_flag("allow_objects", True)
        trans = instantiate_recursive(transform["transform"])
        result.append(Transform(name, transform["deps"], trans))
    return result


def shrink_cached(
    graph: nx.DiGraph,
    transforms: list[Transform],
    cached_step_outs: dict[str, Any],
) -> list[Transform]:
    # Find all steps that already included in cache
    cached_deps = set()
    for node in cached_step_outs:
        cached_deps.update(nx.ancestors(graph, node))

    changed = True
    while changed:
        changed = False
        non_cached = set()
        for cand in cached_deps:
            for edge in graph.edges(cand):
                ancestor = edge[1]
                # there is an edge to non cached step
                if ancestor not in cached_deps and ancestor not in cached_step_outs:
                    # can not remove step since there is a deps to non cached transform
                    non_cached.add(cand)
                    changed = True
        for prev_node in non_cached:
            cached_deps.remove(prev_node)

    shrinked_transforms = []
    idx = 0
    for transform in transforms:
        node = transform.name
        if node in cached_step_outs:
            pipe_step = CachedTransform(cached_step_outs[node])
            new_transform = Transform(node, [], pipe_step)
            shrinked_transforms.append(new_transform)
            graph.nodes[node]["idx"] = idx
            idx += 1
        elif node not in cached_deps:
            shrinked_transforms.append(transform)
            graph.nodes[node]["idx"] = idx
            idx += 1
        else:
            graph.remove_node(node)

    return graph, shrinked_transforms


def create_pipeline_graph(transforms: list[Transform]) -> nx.DiGraph:
    graph = nx.DiGraph()

    # add all nodes
    for idx, transform in enumerate(transforms):
        v = transform.name
        graph.add_node(v, idx=idx)

    # add edges
    for transform in transforms:
        v = transform.name
        for u in transform.deps:
            graph.add_edge(u, v)

    return graph


def execute_pipeline(
    runner: PipelineRunnerBase,
    transforms: list[Transform],
    input_args: DictConfig,
    use_cached: bool,
    cached_step_outs: DictConfig,
) -> None:
    graph = create_pipeline_graph(transforms)
    if not nx.is_directed_acyclic_graph(graph):
        logger.critical("Pipeline has cycle(s)!")
        raise ValueError("Pipeline graph is not a DAG! Please fix a config")

    if use_cached:
        logging.info("Shrink pipeline and reuse cached outputs")
        cache = OmegaConf.to_container(cached_step_outs, resolve=True)
        input_args = OmegaConf.to_container(input_args, resolve=True)
        for key in cache:
            input_args[key] = None

        graph, transforms = shrink_cached(graph, transforms, cache)

    logging.info("Init pipeline")
    runner.init_pipeline(transforms)

    for name in nx.topological_sort(graph):
        transform = transforms[graph.nodes[name]["idx"]]

        logger.info(f"Start {name} pipeline step")

        inputs = runner.get_inputs_for_step(transform)

        if len(transform.deps) == 0:
            inputs = input_args[name]
        elif len(transform.deps) == 1:
            inputs = inputs[0]
        else:
            inputs = tuple(inputs)

        runner.run_step(transform, inputs)


@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
    logging.info("Instantiate pipeline and runner")
    runner = instantiate_recursive(cfg.runner)
    transforms = instantiate_transforms(cfg.pipeline)

    if cfg["log_to_mlflow"]:
        with mlflow.start_run():
            logger.info(
                f"Start MLFlow logging into run: {mlflow.active_run().info.run_id}"
            )

            if "mlflow_tags" in cfg:
                for key, value in cfg["mlflow_tags"].items():
                    mlflow.set_tag(key, value)

            execute_pipeline(
                runner,
                transforms,
                cfg["input_args"],
                cfg["use_cached"],
                cfg["cached_step_outs"],
            )

            mlflow.log_artifact(Path.cwd() / ".hydra/config.yaml")
            if "mlflow_artifacts_paths" in cfg:
                for key, path in cfg["mlflow_artifacts_paths"].items():
                    mlflow.log_artifacts(path, key)
    else:
        execute_pipeline(
            runner,
            transforms,
            cfg["input_args"],
            cfg["use_cached"],
            cfg["cached_step_outs"],
        )


if __name__ == "__main__":
    main()
