#!/usr/bin/env python3

import argparse
import pathlib
import logzero
from logzero import logger
from filelock import FileLock

# globals for release in the end
cuda_devices = []
got_devices_from_folder = False

def _set_loglevel(loglevel):
    if loglevel == "DEBUG":
        logzero.loglevel(logzero.DEBUG)
    elif loglevel == "INFO":
        logzero.loglevel(logzero.INFO)
    elif loglevel == "WARNING":
        logzero.loglevel(logzero.WARNING)
    elif loglevel == "ERROR":
        logzero.loglevel(logzero.ERROR)
    else:
        logger.warning(f"Unknown loglevel {loglevel}, ignoring.")

def _obtain_cuda_devices(number_of_devices, path: pathlib.Path):
    lock_file: pathlib.Path = path / ".lock"
    if not lock_file.exists():
        raise ValueError(f"Invalid GPU directory, lock_file {lock_file} does not exist. Exiting")
    lock = FileLock(lock_file)
    gpus = []
    with lock:
        for i in range(number_of_devices):
            available_gpus = list(path.rglob("*.gpu"))
            if not available_gpus:
                raise Exception("No GPU available! This should never happen.")
            selected_gpu: pathlib.PosixPath = available_gpus[0]
            selected_gpu.unlink()
            gpus.append(int(selected_gpu.stem))
    return gpus

def _release_cuda_devices(cuda_devices, path: pathlib.Path):
    for cuda_device in cuda_devices:
        gpu_file = path / (str(cuda_device) + ".gpu")
        gpu_file.touch(exist_ok=True)

def _train_or_solve(args):
    global cuda_devices
    global got_devices_from_folder

    if args.solver == "gurobi":
        from solvers.gurobi import Gurobi
        solver = Gurobi()
    elif args.solver == "kamis":
        from solvers.kamis import KaMIS
        solver = KaMIS()
    else:
        solver = None

    if solver == None:
        raise ValueError(f"Solver {args.solver} unknown or not implemented yet.")

    if args.pretrained_weights:
        solver.load_weights(args.pretrained_weights)

    if args.operation == "solve":

        if args.cuda_devices:
            cuda_devices = args.cuda_devices
            if args.num_cuda_devices > 0:
                logger.warn(f"Both --cuda_devices and --num_cuda_devices were supplied. Falling back to --cuda_devices = {cuda_devices}!")
        else:
            if args.num_cuda_devices > 0:
                cuda_devices = _obtain_cuda_devices(args.num_cuda_devices, args.cuda_device_folder)
                got_devices_from_folder = True
                logger.info(f"Obtained cuda_devices={cuda_devices} from {args.cuda_device_folder}.")
            else:
                cuda_devices = []
                logger.info("No cuda devices supplied, disabling CUDA.")
                
        parameters = {
            "num_threads": args.num_threads,
            "cuda_devices": cuda_devices,
            "time_limit": args.time_limit,
            "loglevel": args.loglevel
        }

        if args.weighted:
            parameters["weighted"] = "yes"

        if args.self_loops:
            parameters["self_loops"] = "yes"

        if args.reduction:
            parameters["reduction"] = "yes"

        if args.local_search:
            parameters["local_search"] = "yes"

        if args.queue_pruning:
            parameters["queue_pruning"] = "yes"

        if args.max_prob_maps:
            parameters["max_prob_maps"] = args.max_prob_maps

        if args.model_prob_maps:
            parameters["model_prob_maps"] = args.model_prob_maps

        if args.noise_as_prob_maps:
            parameters["noise_as_prob_maps"] = "yes"

        if args.weighted_queue_pop:
            parameters["weighted_queue_pop"] = "yes"

        if args.maximum_iterations_per_episode:
            parameters["maximum_iterations_per_episode"] = args.maximum_iterations_per_episode

        if args.max_nodes:
            parameters["max_nodes"] = args.max_nodes

        if args.quadratic:
            parameters["quadratic"] = "yes"

        if args.write_mps:
            parameters["write_mps"] = "yes"

        if args.prm_file:
            parameters["prm_file"] = args.prm_file

        solver.solve(args.input_folder, args.output_folder, parameters)
    elif args.operation == "train":
        if args.cuda_devices:
            cuda_devices = args.cuda_devices
            if args.num_cuda_devices > 0:
                logger.warn(f"Both --cuda_devices and --num_cuda_devices were supplied. Falling back to --cuda_devices = {cuda_devices}!")
        else:
            if args.num_cuda_devices > 0:
                cuda_devices = _obtain_cuda_devices(args.num_cuda_devices, args.cuda_device_folder)
                got_devices_from_folder = True
                logger.info(f"Obtained cuda_devices={cuda_devices} from {args.cuda_device_folder}.")
            else:
                logger.info("No cuda devices supplied, disabling CUDA.")
        parameters = {
            "cuda_devices": cuda_devices,
            "loglevel": args.loglevel
        }

        if args.weighted:
            parameters["weighted"] = "yes"

        if args.self_loops:
            parameters["self_loops"] = "yes"

        if args.model_prob_maps:
            parameters["model_prob_maps"] = args.model_prob_maps

        if args.epochs:
            parameters["epochs"] = args.epochs

        if args.lr:
            parameters["lr"] = args.lr

        # LwD
        if args.maximum_iterations_per_episode:
            parameters["maximum_iterations_per_episode"] = args.maximum_iterations_per_episode

        if args.num_unrolling_iterations:
            parameters["num_unrolling_iterations"] = args.num_unrolling_iterations

        if args.num_environments_per_batch:
            parameters["num_environments_per_batch"] = args.num_environments_per_batch

        if args.gradient_step_batch_size:
            parameters["gradient_step_batch_size"] = args.gradient_step_batch_size

        if args.gradient_steps_per_update:
            parameters["gradient_steps_per_update"] = args.gradient_steps_per_update

        if args.diversity_reward_coefficient:
            parameters["diversity_reward_coefficient"] = args.diversity_reward_coefficient

        if args.max_entropy_coefficient:
            parameters["max_entropy_coefficient"] = args.max_entropy_coefficient

        if args.num_updates:
            parameters["num_updates"] = args.num_updates

        if args.training_graph_idx:
            parameters["training_graph_idx"] = args.training_graph_idx

        if args.max_nodes:
            parameters["max_nodes"] = args.max_nodes

        solver.train(args.input_folder, args.output_folder, parameters)

    else:
        logger.error(f"Unknown operation: {args.operation}")

    logger.info("Operation done, exiting.")

def _data_generation(args):
    if args.type == "sat":
        from data_generation.sat import SATGraphDataGenerator
        gen = SATGraphDataGenerator(args.input_folder, args.output_folder)
    elif args.type == "random":
        from data_generation.random_graph import RandomGraphGenerator, ErdosRenyi, BarabasiAlbert, HolmeKim, WattsStrogatz, HyperbolicRandomGraph
        if args.model == "er":
            graph_generator = ErdosRenyi(args.min_n, args.max_n, args.er_p)
        elif args.model == "ba":
            graph_generator = BarabasiAlbert(args.min_n, args.max_n, args.ba_m)
        elif args.model == "hk":
            graph_generator = HolmeKim(args.min_n, args.max_n, args.hk_m, args.hk_p)
        elif args.model == "ws":
            graph_generator = WattsStrogatz(args.min_n, args.max_n, args.ws_k, args.ws_p)
        elif args.model == "hrg":
            graph_generator = HyperbolicRandomGraph(args.min_n, args.max_n, args.hrg_alpha, args.hrg_t, args.hrg_degree, args.hrg_threads)
        else:
            raise ValueError(f"Unknown random graph model {args.model}")
        gen = RandomGraphGenerator(args.output_folder, graph_generator, num_graphs=args.num_graphs)
    elif args.type == "realworld":
        from data_generation.realworld import RealWorldGraphGenerator
        limit_rw_graphs = None
        if args.limit_rw_graphs:
            limit_rw_graphs = args.limit_rw_graphs

        gen = RealWorldGraphGenerator(args.output_folder, limit=limit_rw_graphs)
    else:
        raise ValueError(f"Unsupported data type: {args.type}")

    gen.generate(gen_labels=args.gen_labels, weighted=args.weighted)

def main(args):
    ### Set logging mode ###
    _set_loglevel(args.loglevel)

    # Imports are deferred to defer slow DGL import
    if args.operation in ["train", "solve"]:
        _train_or_solve(args)
    elif args.operation == "gendata":
        _data_generation(args)
    else:
        logger.error(f"Unknown operation: {args.operation}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testbench for MIS solvers.")
    subparsers = parser.add_subparsers(help='sub-command help', dest="operation")


    # Global flags
    parser.add_argument("--loglevel", type=str, action="store", default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Verbosity of logging (DEBUG/INFO/WARNING/ERROR)")
    parser.add_argument("--self_loops", action="store_true", default=False, help="Enable self loops addition (in input data) for GCN-based model.")

    train_parser = subparsers.add_parser("train", help="Model training")
    train_parser.add_argument("--weighted", action="store_true", default=False, help="If enabled, solve the weighted MIS problem instead of MIS.")
    train_parser.add_argument("--cuda_devices", type=int, nargs="*", action="store", default=[], help="Which cuda devices should be used (distributed around the threads in round-robin fashion). If not given and --num_cuda_devices is not used, CUDA is disabled.")
    train_parser.add_argument("--num_cuda_devices", type=int, action="store", default=0, help="Alternative to --cuda_devices. Uses a folder to manage available GPUs.")
    train_parser.add_argument("--cuda_device_folder", type=pathlib.Path, action="store", default="/tmp/gpus", help="Folder containing a lockfile for the GPU management. ")
    train_parser.add_argument("--pretrained_weights", type=pathlib.Path, nargs="?", action="store", help="Pre-trained weights to be used for solving/continuing training.")
    train_parser.add_argument("--lr", type=float, action="store", help="Learning rate (for training)")
    train_parser.add_argument("--epochs", type=int, action="store", help="Number of epochs to train for")
    train_parser.add_argument("--model_prob_maps", type=int, action="store", help="Treesearch (Intel/DGL) specific: Number of probability maps the model was/should be trained for.")

    # Lwd Arguments
    train_parser.add_argument("--maximum_iterations_per_episode", type=int, action="store", help="LwD specific: Maximum iterations before the MDP timeouts.")
    train_parser.add_argument("--num_unrolling_iterations", type=int, action="store", help="LwD specific: Maximum number of unrolling iterations (how many stages we have per graph during training).")
    train_parser.add_argument("--num_environments_per_batch", type=int, action="store", help="LwD specific: Graph batch size during training.")
    train_parser.add_argument("--gradient_step_batch_size", type=int, action="store", help="LwD specific: Batch size for gradient step.")
    train_parser.add_argument("--gradient_steps_per_update", type=int, action="store", help="LwD specific: Number of gradient steps per update.")
    train_parser.add_argument("--diversity_reward_coefficient", type=float, action="store", help="LwD specific: Diversity reward coefficient.")
    train_parser.add_argument("--max_entropy_coefficient", type=float, action="store", help="LwD specific: Entropy coefficient.")
    train_parser.add_argument("--num_updates", type=int, action="store", help="LwD specific: How many PPO updates to do.")
    train_parser.add_argument("--training_graph_idx", type=int, action="store", help="LwD specific: On which graph index to continue training.")
    train_parser.add_argument("--max_nodes", type=int, action="store", help="LwD specific: If you have lots of graphs, the determiniation of maximum number of nodes takes some time. If this value is given, you can force-overwrite it to save time.")

    train_parser.add_argument("solver", type=str, help="Solver to use.", choices=["dgl-treesearch", "intel-treesearch", "lwd"])    
    train_parser.add_argument("input_folder", type=pathlib.Path, action="store", help="Directory containing input")
    train_parser.add_argument("output_folder", type=pathlib.Path, action="store",  help="Folder in which the output should be stored (e.g. json containg statistics and solution will be stored, or trained weights)")

    solve_parser = subparsers.add_parser("solve", help="Call a solver")
    solve_parser.add_argument("--time_limit", type=int, nargs="?", action="store", default=600, help="Time limit in seconds")
    solve_parser.add_argument("--num_threads", type=int, nargs="?", action="store", default=8, help="Maximum number of threads to use.")
    solve_parser.add_argument("--weighted", action="store_true", default=False, help="If enabled, solve the weighted MIS problem instead of MIS.")
    solve_parser.add_argument("--cuda_devices", type=int, nargs="*", action="store", default=[], help="Which cuda devices should be used (distributed around the threads in round-robin fashion). If not given, CUDA is disabled.")
    solve_parser.add_argument("--num_cuda_devices", type=int, action="store", default=0, help="Alternative to --cuda_devices. Uses a folder to manage available GPUs.")
    solve_parser.add_argument("--cuda_device_folder", type=pathlib.Path, action="store", default="/tmp/gpus", help="Folder containing a lockfile for the GPU management. ")

    solve_parser.add_argument("--pretrained_weights", type=pathlib.Path, nargs="?", action="store", help="Pre-trained weights to be used for solving/continuing training.")
    solve_parser.add_argument("--reduction", action="store_true", default=False, help="If enabled, reduce graph during tree search.")
    solve_parser.add_argument("--local_search", action="store_true", default=False, help="If enabled, use local_search if time left.")
    solve_parser.add_argument("--queue_pruning", action="store_true", default=False, help="(DGL-Treesearch only) If enabled, prune search queue.")
    solve_parser.add_argument("--noise_as_prob_maps", action="store_true", default=False, help="(DGL-Treesearch and LwD only) If enabled, use uniform noise instead of GNN output.")
    solve_parser.add_argument("--weighted_queue_pop", action="store_true", default=False, help="(DGL-Treesearch only) If enabled, choose element from queue with probability inverse proportional to # of unlabelled vertices in it.")
    solve_parser.add_argument("--max_prob_maps", type=int, action="store", help="DGL-TS specific: number of probability maps to use.")
    solve_parser.add_argument("--model_prob_maps", type=int, action="store", help="Treesearch (Intel/DGL) specific: Number of probability maps the model was/should be trained for.")
    solve_parser.add_argument("--maximum_iterations_per_episode", type=int, action="store", help="LwD specific: Maximum iterations before the MDP timeouts.")
    solve_parser.add_argument("--max_nodes", type=int, action="store", help="LwD specific: If you have lots of graphs, the determiniation of maximum number of nodes takes some time. If this value is given, you can force-overwrite it to save time.")
    solve_parser.add_argument("--quadratic", action="store_true", default=False, help="Gurobi specific: Whether a quadratic program should be used instead of a linear program to solve the MIS problem (cannot be used together with weighted)")
    solve_parser.add_argument("--write_mps", action="store_true", default=False, help="Gurobi specific: Instead of solving, write mps output (e.g., for tuning)")
    solve_parser.add_argument("--prm_file", type=pathlib.Path, nargs="?", action="store", help="Gurboi specific: Gurobi parameter file (e.g. by grbtune).")

    solve_parser.add_argument("solver", type=str, help="Solver to use.", choices=["dgl-treesearch", "intel-treesearch", "gurobi", "kamis", "lwd"])
    solve_parser.add_argument("input_folder", type=pathlib.Path, action="store", help="Directory containing input")
    solve_parser.add_argument("output_folder", type=pathlib.Path, action="store",  help="Folder in which the output should be stored (e.g. json containg statistics and solution will be stored, or trained weights)")

    data_gen_parser = subparsers.add_parser("gendata", help="Generate input data")
    data_gen_parser.add_argument("type", type=str, help="Which data should be generated", choices=["sat", "random", "realworld"])
    data_gen_parser.add_argument("input_folder", type=pathlib.Path, action="store", help="Directory containing input (ignored if type in ['random', 'realworld'])")
    data_gen_parser.add_argument("output_folder", type=pathlib.Path, action="store",  help="Folder in which the output should be stored (e.g. json containg statistics and solution will be stored, or trained weights)")

    data_gen_parser.add_argument("--limit_rw_graphs", type=int, help="[For type = realworld] How many graphs to sample from downloaded realworld datasets")
    data_gen_parser.add_argument("--gen_labels", action="store_true", default=False, help="If enabled, generate labels for graphs using Gurobi.")
    data_gen_parser.add_argument("--weighted", action="store_true", default=False, help="If enabled, generate random vertex weights.")
    data_gen_parser.add_argument("--model", type=str, help="[For type = random] Which random graph model should be used", choices=["er", "ba", "hk", "ws", "hrg"], default="er")
    data_gen_parser.add_argument("--min_n", type=int, help="[For type = random] Minimum number of nodes in the random graph", default=100)
    data_gen_parser.add_argument("--max_n", type=int, help="[For type = random] Maximum number of nodes in the random graph", default=100)
    data_gen_parser.add_argument("--num_graphs", type=int, help="[For type = random] How many graphs to generate", default=10)
    data_gen_parser.add_argument("--er_p", type=float, help="[For type = random and model = er] Parameter p", default=0.5)
    data_gen_parser.add_argument("--ba_m", type=int, help="[For type = random and model = ba] Parameter m", default=10)
    data_gen_parser.add_argument("--hk_m", type=int, help="[For type = random and model = hk] Parameter m", default=10)
    data_gen_parser.add_argument("--hk_p", type=float, help="[For type = random and model = hk] Parameter p", default=0.5)
    data_gen_parser.add_argument("--ws_k", type=int, help="[For type = random and model = ws] Parameter k", default=2)
    data_gen_parser.add_argument("--ws_p", type=float, help="[For type = random and model = w] Parameter p", default=0.5)

    data_gen_parser.add_argument("--hrg_alpha", type=float, help="[For type = random and model = hrg] Parameter Alpha", default=0.75)
    data_gen_parser.add_argument("--hrg_t", type=float, help="[For type = random and model = hrg] Parameter t", default=0)
    data_gen_parser.add_argument("--hrg_degree", type=float, help="[For type = random and model = hrg] Parameter degree", default=2)
    data_gen_parser.add_argument("--hrg_threads", type=float, help="[For type = random and model = hrg] How many threads to use for hyperbolic algorithm", default=8)

    # Optionals without default (defaults are set in the model)
    args = parser.parse_args()
    args.output_folder.mkdir(parents=True, exist_ok=True)

    try:
        main(args)
    except KeyboardInterrupt:
        pass
    finally:
        if got_devices_from_folder:
            logger.info("Releasing cuda devices.")
            _release_cuda_devices(cuda_devices, args.cuda_device_folder)
