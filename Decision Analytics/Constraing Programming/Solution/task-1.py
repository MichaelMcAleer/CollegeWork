"""
Decision Analytics - Assignment 1
Task 1 - Logic Puzzle
Michael McAleer - R00143621

A - Identify the objects for the constraint satisfaction model.
The objects for the model are the people involved:
    - Carol
    - Elisa
    - Oliver
    - Lucas

B - Identify the attributes for the constraint satisfaction model.
The attributes for the model are:
    - Boy, Girl
    - Australia, Canada, South Africa, USA
    - Cambridge, Edinburgh, Oxford, London
    - Architecture, History, Law, Medicine

C - Identify the predicates for the constraint satisfaction model.
The predicates for the model are:
    - is_gender
    - from_origin
    - attends_college
    - studies_course

D - Define the set of variables you need for modelling the puzzle as SAT
problem.
The variables for the model are:
For every person:
    - Gender  [Boy/Girl]                             [True/False] = 4 variables
    - Origin  [Australia, Canada, South Africa, USA] [True/False] = 8 variables
    - College [Cambridge, Edinburgh, Oxford, London] [True/False] = 8 variables
    - Studies [Architecture, History, Law, Medicine] [True/False] = 8 variables
For every person there is 28 variables, for all people there are 112 variables.

E - Define the explicit constraints contained in the sentences of the puzzle
using first order logic.
"One of them is going to London."
    ∃x person(x): college(x, London)

"Exactly one boy and one girl chose a university in a city with the same
initial of their names."
    ∃x ∃y person(x) person(y):
        gender(x, Boy) ∧ gender(y, Boy) ⇒
            (college(x, Oxford) ∧ ¬ college(y, London)) ∨
                (college(y, Oxford) ∧ ¬ college(x, London))

    ∃x ∃y person(x) person(y):
        gender(x, Girl) ∧ gender(y, Girl) ⇒
            (college(x, Cambridge) ∧ ¬ college(y, Edinburgh)) ∨
                (college(y, Cambridge) ∧ ¬ college(x, Edinburgh))

This can be refined in the model to:
    college(Oliver, Oxford) ⇒ ¬ college(Lucas, London)
    college(Lucas, London) ⇒ ¬ college(Oliver, Oxford)
    college(Carol, Cambridge) ⇒ ¬ college(Elisa, Edinburgh)
    college(Elisa, Edinburgh) ⇒ ¬ college(Carol, Cambridge)

"A boy is from Australia, the other studies Architecture."
    ∃x person(x): gender(x, Boy) ∧ origin(x, Australia) ⇒
        ¬ studies(x, Architecture)
    ∃x person(x): gender(x, Boy) ∧ studies(x, Architecture) ⇒
        ¬ origin(x, Australia)

"A girl goes to Cambridge, the other studies Medicine."
    ∃x person(x): gender(x, Girl) ∧ college(x, Cambridge) ⇒
        ¬ studies(x, Medicine)
    ∃x person(x): gender(x, Girl) ∧ studies(x, Medicine) ⇒
        ¬ college(x, Cambridge)

"Oliver studies Law or is from USA. He is not from South Africa."
    (studies(Oliver, Law) ∨ origin(Oliver, USA)) ∧
      ¬(studies(Oliver, Law) ∧ origin(Oliver, USA)) ∧
          ¬ origin(Oliver, South Africa)

"The student from Canada is either a historian or will go to Oxford."
    ∃x person(x): origin(x, Canada) ⇒
        (studies(x, history) ∨ college(x, Oxford)) ∧
            ¬ (studies(x, history) ∧ college(x, Oxford))

"The student from South Africa is going to Edinburgh or will study Law."
    ∃x person(x): origin(x, South Africa) ⇒
        (college(x, Edinburgh) ∨ studies(x, Law)) ∧
            ¬ (college(x, Edinburgh) ∧ studies(x, Law))

F - Define the implicit constraints required for solving the puzzle using first
order logic.
"Every student has a gender/origin/college/course"
    ∀x ∃y person(x) gender(y): gender(x, y)
    ∀x ∃y person(x) origin(y): origin(x, y)
    ∀x ∃y person(x) college(y): college(x, y)
    ∀x ∃y person(x) course(y): course(x, y)

"Every student has no more than one gender/origin/college/course."
    ∀x ∀y ∀z person(x) gender(y) gender(z):
        y ≠ z ⇒ ¬(gender(x, y) ∧ gender(x, z))
    ∀x ∀y ∀z person(x) origin(y) origin(z):
        y ≠ z ⇒ ¬(origin(x, y) ∧ origin(x, z))
    ∀x ∀y ∀z person(x) college(y) college(z):
        y ≠ z ⇒ ¬(college(x, y) ∧ college(x, z))
    ∀x ∀y ∀z person(x) course(y) course(z):
        y ≠ z ⇒ ¬(course(x, y) ∧ course(x, z))

"Every student has a different origin/college/course."
    ∀x ∀y ∀z person(x) person(y) origin(z):
        y ≠ z ⇒ ¬ (origin(x, z) ∧ origin(y, z))
    ∀x ∀y ∀z person(x) person(y) college(z):
        y ≠ z ⇒ ¬ (college(x, z) ∧ college(y, z))
    ∀x ∀y ∀z person(x) person(y) course(z):
        y ≠ z ⇒ ¬ (course(x, z) ∧ course(y, z))

"Carol and Elisa are girls."
    gender(Carol, Girl)
    gender(Elisa, Girl)

"Oliver and Lucas are boys."
    gender(Oliver, Boy)
    gender(Lucas, Boy)
"""
from ortools.sat.python import cp_model

# Define objects
persons = ['Carol', 'Elisa', 'Oliver', 'Lucas']
# Define attributes & predicates
is_gender = ['Boy', 'Girl']
from_origin = ['Australia', 'Canada', 'South Africa', 'USA']
attends_college = ['Cambridge', 'Edinburgh', 'Oxford', 'London']
studies_course = ['Architecture', 'History', 'Law', 'Medicine']


class SolutionPrinter(cp_model.CpSolverSolutionCallback):
    def __init__(self, origin_, college_, course_):
        """Printer for outputting puzzle solutions during solver run.

        :param origin_: origin variables -- list
        :param college_: college variables -- list
        :param course_: course variables -- list
        """
        cp_model.CpSolverSolutionCallback.__init__(self)
        self.origin_ = origin_
        self.college_ = college_
        self.course_ = course_
        self.solutions_ = 0

    def OnSolutionCallback(self):
        """On solution found, print solution details."""
        self.solutions_ = self.solutions_ + 1
        # Print new line between solutions
        if self.solutions_ > 1:
            print()
        print("Solution #{s}:".format(s=self.solutions_))
        for p in persons:
            print(' - {p}:'.format(p=p))
            for o in from_origin:
                if self.Value(self.origin_[p][o]):
                    print('    - Origin  : {o}'.format(o=o))
            for c in attends_college:
                if self.Value(self.college_[p][c]):
                    print('    - College : {c}'.format(c=c))
            for y in studies_course:
                if self.Value(self.course_[p][y]):
                    print('    - Course  : {c}'.format(c=y))

        # J - Answer the question about the nationality of the history student
        for p in persons:
            if self.Value(p_course[p]['History']):
                for c in attends_college:
                    if self.Value(p_college[p][c]):
                        print('Answer: {p} studies History at {c}'.format(
                            p=p, c=c))


if __name__ == "__main__":
    """Run logic puzzle solver."""
    # G - Create a CP-SAT model
    model = cp_model.CpModel()

    # G - Add all variables to the model
    # Gender variables...
    p_gender = dict()
    for person in persons:
        variables = dict()
        for gender in is_gender:
            variables[gender] = model.NewBoolVar(person + gender)
        p_gender[person] = variables
    # Origin variables...
    p_origin = dict()
    for person in persons:
        variables = dict()
        for origin in from_origin:
            variables[origin] = model.NewBoolVar(person + origin)
        p_origin[person] = variables
    # College variables...
    p_college = dict()
    for person in persons:
        variables = dict()
        for college in attends_college:
            variables[college] = model.NewBoolVar(person + college)
        p_college[person] = variables
    # Course variables...
    p_course = dict()
    for person in persons:
        variables = dict()
        for course in studies_course:
            variables[course] = model.NewBoolVar(person + course)
        p_course[person] = variables

    # H - Add all explicit constraints to the CP-SAT model
    # "Exactly one boy and one girl chose a university in a city with the same
    # initial of their names."
    model.AddBoolXOr([
        p_college['Lucas']['London'], p_college['Oliver']['Oxford']])
    model.AddBoolXOr([
        p_college['Carol']['Cambridge'], p_college['Elisa']['Edinburgh']])

    # "A boy is from Australia, the other studies Architecture."
    for person in ['Oliver', 'Lucas']:
        model.AddBoolXOr([
            p_course[person]['Architecture'], p_origin[person]['Australia']])

    # "A girl goes to Cambridge, the other studies Medicine."
    for person in ['Carol', 'Elisa']:
        model.AddBoolXOr([
            p_course[person]['Medicine'], p_college[person]['Cambridge']])

    # "Oliver studies Law or is from USA. He is not from South Africa."
    model.AddBoolXOr([p_course['Oliver']['Law'], p_origin['Oliver']['USA']])
    model.AddBoolAnd([p_origin['Oliver']['South Africa'].Not()])

    # Note: These work without the need of XOR although the implication in the
    # sentence is explicit
    # "The student from Canada is either a historian or will go to Oxford."
    for person in persons:
        model.AddBoolOr([
            p_course[person]['History'],
            p_college[person]['Oxford']]).OnlyEnforceIf(
                p_origin[person]['Canada'])

    # "The student from South Africa is going to Edinburgh or will study Law."
    for person in persons:
        model.AddBoolOr([
            p_course[person]['Law'],
            p_college[person]['Edinburgh']]).OnlyEnforceIf(
                p_origin[person]['South Africa'])

    # I - Add all implicit constraints to the CP-SAT model.
    # Every person has a different...
    for i in range(4):
        for j in range(i + 1, 4):
            for k in range(4):
                # ...Origin
                model.AddBoolOr([
                    p_origin[persons[i]][from_origin[k]].Not(),
                    p_origin[persons[j]][from_origin[k]].Not()])
                # ...College
                model.AddBoolOr([
                    p_college[persons[i]][attends_college[k]].Not(),
                    p_college[persons[j]][attends_college[k]].Not()])
                # ...Course
                model.AddBoolOr([
                    p_course[persons[i]][studies_course[k]].Not(),
                    p_course[persons[j]][studies_course[k]].Not()])

    for person in persons:
        # Every person has at least one...
        # ...Gender
        variables = list()
        for gender in is_gender:
            variables.append(p_gender[person][gender])
        model.AddBoolOr(variables)
        # ...Origin
        variables = list()
        for origin in from_origin:
            variables.append(p_origin[person][origin])
        model.AddBoolOr(variables)
        # ...College
        variables = list()
        for college in attends_college:
            variables.append(p_college[person][college])
        model.AddBoolOr(variables)
        # ...Course
        variables = list()
        for course in studies_course:
            variables.append(p_course[person][course])
        model.AddBoolOr(variables)

        # Every person has at most one...
        for i in range(4):
            for j in range(i + 1, 4):
                # ...Origin
                model.AddBoolOr([
                    p_origin[person][from_origin[i]].Not(),
                    p_origin[person][from_origin[j]].Not()])
                # ...College
                model.AddBoolOr([
                    p_college[person][attends_college[i]].Not(),
                    p_college[person][attends_college[j]].Not()])
                # ...Course
                model.AddBoolOr([
                    p_course[person][studies_course[i]].Not(),
                    p_course[person][studies_course[j]].Not()])
        for i in range(2):
            for j in range(i + 1, 2):
                # ...Gender
                model.AddBoolOr([
                    p_gender[person][is_gender[i]].Not(),
                    p_gender[person][is_gender[j]].Not()])

    # Carol and Elisa are girls
    model.AddBoolAnd([p_gender['Carol']['Girl']])
    model.AddBoolAnd([p_gender['Elisa']['Girl']])

    # Oliver and Lucas are boys
    model.AddBoolAnd([p_gender['Oliver']['Boy']])
    model.AddBoolAnd([p_gender['Lucas']['Boy']])

    # J - Solve the CP-SAT model
    solver = cp_model.CpSolver()
    solver.SearchForAllSolutions(
        model, SolutionPrinter(p_origin, p_college, p_course))

    # The answer of who is the history student is output in the solution
    # printer as there is more than one solution