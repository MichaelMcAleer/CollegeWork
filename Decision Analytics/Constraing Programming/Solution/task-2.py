"""
Decision Analytics - Assignment 1
Task 2 - Project Planning
Michael McAleer - R00143621

A - Define the set of variables you need for modelling this constrained
optimisation problem.
Variables required for the constrained optimisation problem are:
For every project:
    - Project selected [True/False] = 2 variables
    - Project value (int)           = 1 variable
    - Project cost (int)            = 1 variable
For every project there is 4 variables, for all projects there are 40
variables.

B - Define the constraints to model the incompatibilities between projects.
- Project 4 is incompatible with project 1
- Project 7 is incompatible with project 1 or project 2
- Project 10 is incompatible with project 4
The equivalent logic for project incompatibilities:
    ∃x ∃y: project(x) ⇒ ¬ project(y)

C - Define the constraints to model the prerequisites of projects.
- Project 3 requires project 1 and project 2
- Project 5 requires project 3 and project 4
- Project 8 requires project 3 and project 6
- Project 9 requires project 3 and project 5
The equivalent logic for project requirements:
    ∃x ∃y ∃z: project(x) ⇒ project(y) ∧ project(z)
    
D - Define the constraint to model the overall budget restriction.
The budget restriction constraint is the sum of all selected projects' cost
must not exceed 400. The equivalent logic for cost constraint:
    sum(selected_projects[cost]) <= 400

E - Define the maximisation criterion for the problem.
The maximisation criterion for project selection is to maximise the sum of
projects' value whilst not exceeding the overall budget restriction of 400.
The equivalent logic for project value maximisation:
    max(sum(selected_projects[value])) while (
        sum(selected_projects[cost]) <= 400)
"""
from ortools.sat.python import cp_model


class SolPrint(cp_model.CpSolverSolutionCallback):
    def __init__(self, projects_in, project_vars_in, total_cost_in,
                 total_value_in):
        """Printer for outputting feasible solutions during solver run.
        
        :param projects_in: project definition -- list
        :param project_vars_in: model variables -- list
        :param total_cost_in: total selected projects cost -- int
        :param total_value_in: total selected projects value -- int
        """
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.p_ = projects_in
        self.p_vars_ = project_vars_in
        self.total_cost_ = total_cost_in
        self.total_value_ = total_value_in
        self.total_plans = 0

    def OnSolutionCallback(self):
        """On solution found, print solution details."""
        self.total_plans += 1
        print('Feasible Project Plan #{c}:'.format(c=self.total_plans))
        for idx in range(0, len(self.p_)):
            if self.Value(self.p_vars_[idx]):
                print('  - Project ID: {p} (Cost={c}, Value={v})'.format(
                    p=(idx + 1), c=self.p_[idx][4], v=self.p_[idx][3]))
        print('  - Total Cost  : {c}'.format(c=self.Value(self.total_cost_)))
        print('  - Total Value : {v}'.format(v=self.Value(self.total_value_)))


def get_project_index(tgt_id):
    """Given a project id, find the related project index in project list. 
    
    :param tgt_id: target project id -- int
    :return: target project index -- int
    """
    return tgt_id - 1


if __name__ == "__main__":
    """Run project planner."""
    # Project Tuple Map:
    #   i0: ID | i1: Requires | i2: Incompatible | i3: Value | i4: Cost
    projects = [(1, [], [], 18, 12),
                (2, [], [], 51, 43),
                (3, [1, 2], [], 32, 12),
                (4, [], [1], 80, 76),
                (5, [3, 4], [], 65, 42),
                (6, [], [], 44, 43),
                (7, [], [1, 2], 91, 87),
                (8, [3, 6], [], 65, 43),
                (9, [3, 5], [], 92, 65),
                (10, [], [4], 69, 62)]
    
    # F - Create a CP-SAT model in Python
    model = cp_model.CpModel()
    # Instantiate CP-SAT model and required lists to hold project variables,
    # projects' cost, and projects' value
    project_vars = list()
    cost_project_vars = list()
    value_project_vars = list()

    # F - Add all variables to the CP-SAT model
    for i in range(0, len(projects)):
        project_id = 'project_{n}'.format(n=i+1)
        project_vars.append(model.NewBoolVar(project_id))
        cost_project_vars.append(projects[i][4] * project_vars[i])
        value_project_vars.append(projects[i][3] * project_vars[i])

    for project in projects:
        # G - Add all project incompatibility constraints to the CP-SAT model
        if project[2]:
            for incompatibility in project[2]:
                project_id = get_project_index(project[0])
                incompatible_id = get_project_index(incompatibility)
                model.AddBoolAnd([
                    project_vars[project_id].Not()]).OnlyEnforceIf([
                        project_vars[incompatible_id]])

        # H - Add all prerequisite constraints to the CP-SAT model
        if project[1]:
            clause = list()
            project_id = get_project_index(project[0])
            # For each pre-requisite project add them to a list to be used as
            # the clause in the AND condition and only enforced if the
            # selected project is TRUE
            for requirement in project[1]:
                requirement_id = get_project_index(requirement)
                clause.append(project_vars[requirement_id])
            model.AddBoolAnd(clause).OnlyEnforceIf(project_vars[project_id])

    # Calculate the total cost and value of selected projects
    total_cost = sum(cost_project_vars)
    total_value = sum(value_project_vars)
    # Set the project budget constraint value
    project_cost_limit = 400
    # I - Add the budget constraint to the CP-SAT model
    model.Add(total_cost <= project_cost_limit)
    # Instantiate the CP-SAT solver
    solver = cp_model.CpSolver()
    # Search for all possible solutions given the model constraints
    solver.SearchForAllSolutions(
        model, SolPrint(projects, project_vars, total_cost, total_value))
    # J - Add the maximisation constraint to the CP-SAT model
    model.Maximize(total_value)
    # K - Solve the CP-SAT model
    status = solver.Solve(model)
    print('\nSolution Status: {s}'.format(s=solver.StatusName(status)))
    # K - Output which projects to select
    print('{s} Project Plan:'.format(s=solver.StatusName(status)))
    for i in range(0, len(projects)):
        if solver.Value(project_vars[i]):
            print('  - Project ID: {p} (Cost={c}, Value={v})'.format(
                p=(i + 1), c=projects[i][4], v=projects[i][3]))
    # K - Calculate the investment required and the profit generated
    print('Total Investment Required : {c}'.format(
        c=solver.Value(total_cost)))
    print('Total Value Generated     : {v}'.format(
        v=solver.Value(total_value)))
    print('Total Profit Generated    : {p}'.format(
        p=(solver.Value(total_value) - solver.Value(total_cost))))
    # Output additional solution stats
    print('\nSolution Statistics:')
    print('  - Conflicts : {c}'.format(c=solver.NumConflicts()))
    print('  - Branches  : {b}'.format(b=solver.NumBranches()))
    print('  - Wall time : {t}s'.format(t=solver.WallTime()))
