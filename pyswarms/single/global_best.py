# -*- coding: utf-8 -*-

r"""
A Global-best Particle Swarm Optimization (gbest PSO) algorithm.

It takes a set of candidate solutions, and tries to find the best
solution using a position-velocity update method. Uses a
star-topology where each particle is attracted to the best
performing particle.

The position update can be defined as:

.. math::

   x_{i}(t+1) = x_{i}(t) + v_{i}(t+1)

Where the position at the current timestep :math:`t` is updated using
the computed velocity at :math:`t+1`. Furthermore, the velocity update
is defined as:

.. math::

   v_{ij}(t + 1) = w * v_{ij}(t) + c_{1}r_{1j}(t)[y_{ij}(t) − x_{ij}(t)]
                   + c_{2}r_{2j}(t)[\hat{y}_{j}(t) − x_{ij}(t)]

Here, :math:`c1` and :math:`c2` are the cognitive and social parameters
respectively. They control the particle's behavior given two choices: (1) to
follow its *personal best* or (2) follow the swarm's *global best* position.
Overall, this dictates if the swarm is explorative or exploitative in nature.
In addition, a parameter :math:`w` controls the inertia of the swarm's
movement.

An example usage is as follows:

.. code-block:: python

    import pyswarms as ps
    from pyswarms.utils.functions import single_obj as fx

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

    # Call instance of GlobalBestPSO
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
                                        options=options)

    # Perform optimization
    stats = optimizer.optimize(fx.sphere, iters=100)

This algorithm was adapted from the earlier works of J. Kennedy and
R.C. Eberhart in Particle Swarm Optimization [IJCNN1995]_.

.. [IJCNN1995] J. Kennedy and R.C. Eberhart, "Particle Swarm Optimization,"
    Proceedings of the IEEE International Joint Conference on Neural
    Networks, 1995, pp. 1942-1948.
"""

# Import standard library
import json
import logging
import os

# Import modules
import numpy as np
import multiprocessing as mp

from collections import deque

from ..backend.operators import compute_pbest, compute_objective_function
from ..backend.topology import Star
from ..backend.handlers import BoundaryHandler, VelocityHandler, OptionsHandler
from ..backend.swarms import Swarm


from ..base import SwarmOptimizer
from ..utils.reporter import Reporter


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class GlobalBestPSO(SwarmOptimizer):
    def __init__(
        self,
        n_particles,
        dimensions,
        options,
        bounds=None,
        oh_strategy=None,
        bh_strategy="periodic",
        velocity_clamp=None,
        vh_strategy="unmodified",
        center=1.00,
        ftol=-np.inf,
        ftol_iter=1,
        init_pos=None,
        checkpoint_interval=20,
        checkpoint="__checkpoint__.json",
    ):
        """Initialize the swarm

        Attributes
        ----------
        n_particles : int
            number of particles in the swarm.
        dimensions : int
            number of dimensions in the space.
        options : dict with keys :code:`{'c1', 'c2', 'w'}`
            a dictionary containing the parameters for the specific
            optimization technique.
                * c1 : float
                    cognitive parameter
                * c2 : float
                    social parameter
                * w : float
                    inertia parameter
        bounds : tuple of numpy.ndarray, optional
            a tuple of size 2 where the first entry is the minimum bound while
            the second entry is the maximum bound. Each array must be of shape
            :code:`(dimensions,)`.
        oh_strategy : dict, optional, default=None(constant options)
            a dict of update strategies for each option.
        bh_strategy : str
            a strategy for the handling of out-of-bounds particles.
        velocity_clamp : tuple, optional
            a tuple of size 2 where the first entry is the minimum velocity and
            the second entry is the maximum velocity. It sets the limits for
            velocity clamping.
        vh_strategy : str
            a strategy for the handling of the velocity of out-of-bounds particles.
        center : list (default is :code:`None`)
            an array of size :code:`dimensions`
        ftol : float
            relative error in objective_func(best_pos) acceptable for
            convergence. Default is :code:`-np.inf`
        ftol_iter : int
            number of iterations over which the relative error in
            objective_func(best_pos) is acceptable for convergence.
            Default is :code:`1`
        init_pos : numpy.ndarray, optional
            option to explicitly set the particles' initial positions. Set to
            :code:`None` if you wish to generate the particles randomly.
        """
        if os.path.isfile(checkpoint):
            self.checkpoint_interval = checkpoint_interval
            self._from_checkpoint(checkpoint=checkpoint)
        else:
            super(GlobalBestPSO, self).__init__(
                n_particles=n_particles,
                dimensions=dimensions,
                options=options,
                bounds=bounds,
                velocity_clamp=velocity_clamp,
                center=center,
                ftol=ftol,
                ftol_iter=ftol_iter,
                init_pos=init_pos,
            )

            if oh_strategy is None:
                oh_strategy = {}
            # Initialize logger
            self.rep = Reporter(logger=logging.getLogger(__name__))
            # Initialize the resettable attributes
            self.reset()
            # Initialize the topology
            self.top = Star()
            self.bh = BoundaryHandler(strategy=bh_strategy)
            self.vh = VelocityHandler(strategy=vh_strategy)
            self.oh = OptionsHandler(strategy=oh_strategy)
            self.name = __name__
            self.checkpoint_interval = checkpoint_interval
            self.starting_iter = 0
            self.restore = False
            self.checkpoint_file = None

        self.rep.log(f"Setting checkpoint interval to {self.checkpoint_interval}",lvl=logging.INFO)

    def checkpoint(self, current_iteration):
        """Checkpoint the state to a JSON file.
        """
        optimizer = {}
        # First, store all the things one would pass to the constructor
        optimizer.update(n_particles=self.n_particles)
        optimizer.update(dimensions=self.dimensions)
        optimizer.update(options=self.options)
        optimizer.update(bh_strategy=self.bh.strategy)
        optimizer.update(vh_strategy=self.vh.strategy)
        optimizer.update(oh_strategy=self.oh.strategy)
        optimizer.update(bounds=self.bounds)
        optimizer.update(velocity_clamp=self.velocity_clamp)
        optimizer.update(ftol=self.ftol)
        optimizer.update(ftol_iter=self.ftol_iter)
        optimizer.update(center=self.center)
        # Some information on the current iteration
        optimizer.update(current_iteration=current_iteration)

        # Now store current properties of the swarm
        optimizer.update(position=self.swarm.position)
        optimizer.update(velocity=self.swarm.velocity)
        optimizer.update(current_cost=self.swarm.current_cost)
        optimizer.update(pbest_pos=self.swarm.pbest_pos)
        optimizer.update(pbest_cost=self.swarm.pbest_cost)
        optimizer.update(best_pos=self.swarm.best_pos)
        optimizer.update(best_cost=self.swarm.best_cost)
        # The swarm options
        optimizer.update(swarm_options=self.swarm.options)
        # Store the histories so they can be repopulated
        optimizer.update(cost_history=self.cost_history)
        optimizer.update(mean_pbest_history=self.mean_pbest_history)
        optimizer.update(mean_neighbor_history=self.mean_neighbor_history)
        optimizer.update(pos_history=self.pos_history)
        optimizer.update(velocity_history=self.velocity_history)
        with open("__checkpoint__.json", "w") as fw:
            json.dump(
                optimizer, fw, cls=NumpyEncoder, indent=4, sort_keys=True
            )

    def _from_checkpoint(self, checkpoint="__checkpoint__.json"):
        """Initialize the optimizer and swarm from checkpoint saved to disk

        Args:
            checkpoint (str): The checkpoint file
        """
        # Read the checkpoint file
        with open(checkpoint) as fp:
            optimizer = json.load(fp)
        # Call the parent constructor
        # Note that this will create a new swarm

        super(GlobalBestPSO, self).__init__(
            n_particles=optimizer["n_particles"],
            dimensions=optimizer["dimensions"],
            options=optimizer["options"],
            bounds=optimizer["bounds"],
            velocity_clamp=optimizer["velocity_clamp"],
            center=optimizer["center"],
            ftol=optimizer["ftol"],
            ftol_iter=optimizer["ftol_iter"],
        )

        # Restore the swarm
        self.swarm = Swarm(
            position=np.array(optimizer["position"]),
            velocity=np.array(optimizer["velocity"]),
            pbest_pos=np.array(optimizer["pbest_pos"]),
            pbest_cost=np.array(optimizer["pbest_cost"]),
            options=optimizer["swarm_options"],
            best_pos=np.array(optimizer["best_pos"]),
            best_cost=np.float64(optimizer["best_cost"]),
            current_cost=np.array(optimizer["current_cost"]),
        )

        # Restore various strategies
        self.bh = BoundaryHandler(strategy=optimizer["bh_strategy"])
        self.vh = VelocityHandler(strategy=optimizer["vh_strategy"])
        self.oh = OptionsHandler(strategy=optimizer["oh_strategy"])

        # Re-initialize the histories
        # Initialize logger
        self.rep = Reporter(logger=logging.getLogger(__name__))
        # Initialize the topology
        self.top = Star()
        self.name = __name__
        self.starting_iter = optimizer["current_iteration"]+1
        self.restore = True
        self.checkpoint_file = checkpoint
       

    def optimize(
        self, objective_func, iters, n_processes=None, verbose=True, **kwargs
    ):
        """Optimize the swarm for a number of iterations

        Performs the optimization to evaluate the objective
        function :code:`f` for a number of iterations :code:`iter.`

        Parameters
        ----------
        objective_func : callable
            objective function to be evaluated
        iters : int
            number of iterations
        n_processes : int
            number of processes to use for parallel particle evaluation (default: None = no parallelization)
        verbose : bool
            enable or disable the logs and progress bar (default: True = enable logs)
        kwargs : dict
            arguments for the objective function

        Returns
        -------
        tuple
            the global best cost and the global best position.
        """

        # Apply verbosity
        if verbose:
            log_level = logging.INFO
        else:
            log_level = logging.NOTSET

        self.rep.log("Obj. func. args: {}".format(kwargs), lvl=logging.DEBUG)
        self.rep.log(
            "Optimize for {} total iters with {}".format(iters, self.options),
            lvl=log_level,
        )
        # Populate memory of the handlers
        self.bh.memory = self.swarm.position
        self.vh.memory = self.swarm.position

        # Setup Pool of processes for parallel evaluation
        pool = None if n_processes is None else mp.Pool(n_processes)
        remaining_iters = iters - self.starting_iter
        if not self.restore:
            self.swarm.pbest_cost = np.full(self.swarm_size[0], np.inf)
        else:
            self.rep.log(f"Resuming from iteration {self.starting_iter}, {remaining_iters} left", lvl=log_level)
        ftol_history = deque(maxlen=self.ftol_iter)
        for i in self.rep.pbar(remaining_iters, self.name) if verbose else range(remaining_iters):
            # Compute cost for current position and personal best
            # fmt: off
            self.swarm.current_cost = compute_objective_function(self.swarm, objective_func, pool=pool, **kwargs)
            self.swarm.pbest_pos, self.swarm.pbest_cost = compute_pbest(self.swarm)
            # Set best_cost_yet_found for ftol
            best_cost_yet_found = self.swarm.best_cost
            self.swarm.best_pos, self.swarm.best_cost = self.top.compute_gbest(self.swarm)
            # fmt: on
            if verbose:
                self.rep.hook(best_cost=self.swarm.best_cost)
            # Save to history
            hist = self.ToHistory(
                best_cost=self.swarm.best_cost,
                mean_pbest_cost=np.mean(self.swarm.pbest_cost),
                mean_neighbor_cost=self.swarm.best_cost,
                position=self.swarm.position,
                velocity=self.swarm.velocity,
            )
            self._populate_history(hist)
            # Verify stop criteria based on the relative acceptable cost ftol
            relative_measure = self.ftol * (1 + np.abs(best_cost_yet_found))
            delta = (
                np.abs(self.swarm.best_cost - best_cost_yet_found)
                < relative_measure
            )
            if i < self.ftol_iter:
                ftol_history.append(delta)
            else:
                ftol_history.append(delta)
                if all(ftol_history):
                    break
            # Perform options update
            self.swarm.options = self.oh(
                self.options, iternow=i+self.starting_iter, itermax=iters
            )
            # Perform velocity and position updates
            self.swarm.velocity = self.top.compute_velocity(
                self.swarm, self.velocity_clamp, self.vh, self.bounds
            )
            self.swarm.position = self.top.compute_position(
                self.swarm, self.bounds, self.bh
            )
            # If checkpointing is requested, pack up all the info needed
            # to restore to a json file
            if self.checkpoint_interval is not None:
                if (i+1) % self.checkpoint_interval == 0:
                    self.rep.log(f"Checkpointing at iteration {i+self.starting_iter}",lvl=log_level)
                    self.checkpoint(i+self.starting_iter)
        # Obtain the final best_cost and the final best_position
        
        final_best_cost = self.swarm.best_cost.copy()
        final_best_pos = self.swarm.pbest_pos[
            self.swarm.pbest_cost.argmin()
        ].copy()
        # Write report in log and return final cost and position
        self.rep.log(
            "Optimization finished | best cost: {}, best pos: {}".format(
                final_best_cost, final_best_pos
            ),
            lvl=log_level,
        )
        # Close Pool of Processes
        if n_processes is not None:
            pool.close()

        if self.restore:
            if os.path.isfile(self.checkpoint_file):
                os.remove(self.checkpoint_file)
        return (final_best_cost, final_best_pos)
