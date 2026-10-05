from copy import deepcopy
from typing import List, Tuple, Any, Union

import pandas as pd



class ParetoFront:
    """
    A data structure for maintaining Pareto-optimal solutions in multi-objective optimization.

    Stores solutions and their corresponding objective vectors, automatically managing
    the Pareto frontier by adding non-dominated solutions and optionally removing
    dominated ones.
    """

    def __init__(self, store_all: bool = False, minimize: Union[bool, List[bool]] = True):
        """
        Initialize the ParetoFront.

        Args:
            store_all: If True, keep all solutions ever added. If False, only keep Pareto-optimal ones.
            minimize: If bool, applies to all objectives. If list of bools, specifies per objective
                     whether to minimize (True) or maximize (False).
        """
        self.store_all = store_all
        self.minimize = minimize

        # Core data structure for Pareto-optimal solutions
        self.pareto_solutions: List[Tuple[Any, Tuple]] = []

        # Additional storage for all solutions if requested
        if store_all:
            self.all_solutions: List[Tuple[Any, Tuple]] = []

        # Cache for objective dimensionality (set on first add)
        self._objective_dim = None

    def add(self, solution: Any, objective_vector: Tuple) -> bool:
        """
        Add a solution with its objective vector to the Pareto front.

        Args:
            solution: The solution object (can be any type); stored by reference, not copied
            objective_vector: Tuple of objective function values

        Returns:
            bool: True if solution was added to Pareto front, False if dominated
        """
        if not isinstance(objective_vector, (tuple, list)):
            raise ValueError("objective_vector must be a tuple or list")

        objective_vector = tuple(objective_vector)  # Ensure it's a tuple

        # Initialize objective dimensionality on first solution
        if self._objective_dim is None:
            self._objective_dim = len(objective_vector)
            # Convert minimize to list format for easier processing
            if isinstance(self.minimize, bool):
                self.minimize = [self.minimize] * self._objective_dim
            elif len(self.minimize) != self._objective_dim:
                raise ValueError("Length of minimize list must match objective vector dimension")

        if len(objective_vector) != self._objective_dim:
            raise ValueError(f"Objective vector dimension mismatch. Expected {self._objective_dim}, got {len(objective_vector)}")

        # Check if new solution is dominated by any existing Pareto solution
        if self._is_dominated_by_any(objective_vector):
            # Add to all_solutions if storing all, but not to Pareto front
            if self.store_all:
                self.all_solutions.append((solution, objective_vector))
            return False

        # Solution is not dominated, so it joins the Pareto front
        self.pareto_solutions.append((solution, objective_vector))

        # Remove any existing solutions that are now dominated by this new solution
        self._remove_dominated_by(objective_vector)

        # Add to all_solutions if storing all
        if self.store_all:
            self.all_solutions.append((solution, objective_vector))

        return True

    def _dominates(self, obj_vec1: Tuple, obj_vec2: Tuple) -> bool:
        """
        Check if obj_vec1 dominates obj_vec2.

        obj_vec1 dominates obj_vec2 if:
        - obj_vec1 is better or equal in all objectives, AND
        - obj_vec1 is strictly better in at least one objective
        """
        better_or_equal_all = True
        strictly_better_any = False

        for i, (val1, val2) in enumerate(zip(obj_vec1, obj_vec2)):
            if self.minimize[i]:  # Minimization objective
                if val1 > val2:  # val1 is worse
                    better_or_equal_all = False
                    break
                elif val1 < val2:  # val1 is strictly better
                    strictly_better_any = True
            else:  # Maximization objective
                if val1 < val2:  # val1 is worse
                    better_or_equal_all = False
                    break
                elif val1 > val2:  # val1 is strictly better
                    strictly_better_any = True

        return better_or_equal_all and strictly_better_any

    def _is_dominated_by_any(self, new_objective_vector: Tuple) -> bool:
        """Check if the new objective vector is dominated by any existing Pareto solution."""
        for _, existing_obj_vec in self.pareto_solutions:
            if self._dominates(existing_obj_vec, new_objective_vector):
                return True
        return False

    def _remove_dominated_by(self, new_objective_vector: Tuple) -> None:
        """Remove all existing Pareto solutions that are dominated by the new objective vector."""
        # Build new list of non-dominated solutions for efficiency
        self.pareto_solutions = [
            (sol, obj_vec) for sol, obj_vec in self.pareto_solutions
            if not self._dominates(new_objective_vector, obj_vec)
        ]

    def _objective_vectors_equal(self, obj_vec1: Tuple, obj_vec2: Tuple) -> bool:
        """Check if two objective vectors are equal."""
        return obj_vec1 == obj_vec2

    def _to_dataframe(self, solutions_list: List[Tuple[Any, Tuple]]) -> pd.DataFrame:
        """Convert a list of (solution, objective_vector) tuples to a DataFrame."""
        if not solutions_list:
            # Return empty DataFrame with appropriate columns
            columns = ['solution'] + [f'objective_{i}' for i in range(self._objective_dim or 0)]
            return pd.DataFrame(columns=columns)

        # Extract solutions and objective vectors
        solutions = [sol for sol, _ in solutions_list]
        objectives = [obj_vec for _, obj_vec in solutions_list]

        # Create DataFrame
        data = {'solution': solutions}
        for i in range(len(objectives[0])):
            data[f'objective_{i}'] = [obj_vec[i] for obj_vec in objectives]

        return pd.DataFrame(data)

    # Utility methods for accessing data

    def get_pareto_front(self, as_dataframe: bool = False) -> Union[List[Tuple[Any, Tuple]], pd.DataFrame]:
        """Return list of (solution, objective_vector) pairs on the Pareto front."""
        if as_dataframe:
            return self._to_dataframe(self.pareto_solutions)
        return self.pareto_solutions.copy()

    def get_all_solutions(self, as_dataframe: bool = False) -> Union[List[Tuple[Any, Tuple]], pd.DataFrame]:
        """
        Return list of all (solution, objective_vector) pairs ever added.
        Only available if store_all=True was set in constructor.
        """
        if not self.store_all:
            raise ValueError("get_all_solutions() only available when store_all=True")

        if as_dataframe:
            return self._to_dataframe(self.all_solutions)
        return self.all_solutions.copy()

    def get_pareto_objectives(self) -> List[Tuple]:
        """Return just the objective vectors of Pareto-optimal solutions."""
        return [obj_vec for _, obj_vec in self.pareto_solutions]

    def get_pareto_solutions(self) -> List[Any]:
        """Return just the solution objects of Pareto-optimal solutions."""
        return [solution for solution, _ in self.pareto_solutions]

    def pareto_size(self) -> int:
        """Return the number of solutions on the Pareto front."""
        return len(self.pareto_solutions)

    def total_size(self) -> int:
        """Return the total number of solutions stored."""
        if self.store_all:
            return len(self.all_solutions)
        else:
            return len(self.pareto_solutions)

    def clear(self) -> None:
        """Remove all solutions from the data structure."""
        self.pareto_solutions.clear()
        if self.store_all:
            self.all_solutions.clear()
        self._objective_dim = None

    def is_empty(self) -> bool:
        """Check if the Pareto front is empty."""
        return len(self.pareto_solutions) == 0

    def __len__(self) -> int:
        """Return the number of solutions on the Pareto front."""
        return self.pareto_size()

    def __str__(self) -> str:
        """String representation of the ParetoFront."""
        return f"ParetoFront(pareto_size={self.pareto_size()}, total_size={self.total_size()})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"ParetoFront(store_all={self.store_all}, minimize={self.minimize}, pareto_size={self.pareto_size()})"