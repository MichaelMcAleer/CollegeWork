# Decision Analytics
# Assignment 2: Linear Programming - Part 1
# Michael McAleer - R00143621
import pandas as pd
from ortools.linear_solver import pywraplp

"""
A - Load the input data from the file 'Assignment_DA_2_a_data.xlsx'
"""
raw_data = 'Assignment_DA_2_a_data.xlsx'

supplier_stock = pd.read_excel(
    raw_data, sheet_name='Supplier stock', index_col=0)

raw_material_cost = pd.read_excel(
    raw_data, sheet_name='Raw material costs', index_col=0)

raw_material_shipping = pd.read_excel(
    raw_data, sheet_name='Raw material shipping', index_col=0)

product_requirements = pd.read_excel(
    raw_data, sheet_name='Product requirements', index_col=0)

production_capacity = pd.read_excel(
    raw_data, sheet_name='Production capacity', index_col=0)

production_cost = pd.read_excel(
    raw_data, sheet_name='Production cost', index_col=0)

shipping_costs = pd.read_excel(
    raw_data, sheet_name='Shipping costs', index_col=0)

customer_demand = pd.read_excel(
    raw_data, sheet_name='Customer demand', index_col=0)

# Extract actors from dataset into sets of unique values
customers = list(sorted(customer_demand.columns))
products = list(sorted(customer_demand.index))
factories = list(sorted(production_cost.columns))
suppliers = list(sorted(supplier_stock.index))
materials = list(sorted(supplier_stock.columns))

"""
B - Identify the decision variables of the problem for the Linear Programming 
model.

    - We need to decide how much to supply from each of the three factories to
    each of the four customers. 
        
        Let X^ijk be the number of products i (i={1,2,3,4}) supplied from 
        factory j (j={1,2,3}) to customer k (k={1,2,3,4})

        -------      -------------------------      ------
        Fact         pA   | pB   | pC   | pD        Cust
        -------      -------------------------      ------
        Fact A   |   X111 + X211 + X311 + X411   |  Cust A
        Fact A   |   X112 + X212 + X312 + X412   |  Cust B
        Fact A   |   X113 + X213 + X313 + X413   |  Cust C
        Fact A   |   X114 + X214 + X314 + X414   |  Cust D
        -------      -------------------------      ------
        Fact B   |   X121 + X221 + X321 + X421   |  Cust A
        Fact B   |   X122 + X222 + X322 + X422   |  Cust B
        Fact B   |   X123 + X223 + X323 + X423   |  Cust C
        Fact B   |   X124 + X224 + X324 + X424   |  Cust D
        -------      -------------------------      ------
        Fact C   |   X131 + X231 + X331 + X431   |  Cust A
        Fact C   |   X132 + X232 + X332 + X432   |  Cust B
        Fact C   |   X133 + X233 + X333 + X433   |  Cust C
        Fact C   |   X134 + X234 + X334 + X434   |  Cust D
        -------      -------------------------      ------

    - We need to decide what suppliers factories will order materials from.
        
        Let Y^abc be factory a (a={1,2,3}) ordering material b (b={1,2,3,4})
        from supplier c (c={1,2,3,4,5})

        -------      -------------------------      --------
        Fact         mA   | mB   | mC   | mD        Supplier
        -------      -------------------------      --------
        Fact A   |   Y111 + Y211 + Y311 + Y411   |  Sup A
        Fact A   |   Y112 + Y212 + Y312 + Y412   |  Sup B
        Fact A   |   Y113 + Y213 + Y313 + Y413   |  Sup C
        Fact A   |   Y114 + Y214 + Y314 + Y414   |  Sup D
        Fact A   |   Y115 + Y215 + Y315 + Y415   |  Sup E
        -------      -------------------------      --------
        Fact B   |   Y121 + Y221 + Y321 + Y421   |  Sup A
        Fact B   |   Y122 + Y222 + Y322 + Y422   |  Sup B
        Fact B   |   Y123 + Y223 + Y323 + Y423   |  Sup C
        Fact B   |   Y124 + Y224 + Y324 + Y424   |  Sup D
        Fact B   |   Y125 + Y225 + Y325 + Y425   |  Sup E
        -------      -------------------------      --------
        Fact C   |   Y131 + Y231 + Y331 + Y431   |  Sup A
        Fact C   |   Y132 + Y232 + Y332 + Y432   |  Sup B
        Fact C   |   Y133 + Y233 + Y333 + Y433   |  Sup C
        Fact C   |   Y134 + Y234 + Y334 + Y434   |  Sup D
        Fact C   |   Y135 + Y235 + Y335 + Y435   |  Sup E
        -------      -------------------------      --------

C - Identify the constraints of the problem for the Linear Programming model.

    - Each customer needs to have their demand met
        Total sum of all factories per product must be equal to customer demand
            --------------------------
            customer A demands
            --------------------------
            pA: X111 + X121 + X131 = 7
            pD: X411 + X421 + X431 = 1
            --------------------------
            customer B demands
            --------------------------
            pA: X112 + X122 + X132 = 3
            --------------------------
            customer C demands
            --------------------------
            pB: X213 + X223 + X233 = 2
            pD: X413 + X423 + X433 = 3
            --------------------------
            customer D demands
            --------------------------
            pC: X314 + X324 + X334 = 4
            pD: X414 + X424 + X434 = 4
            --------------------------
            
    - Factories have limited production capacity
        
        For all factories, sum of all products produced across per customer 
        must not not exceed the factory production limit for that product.
        
            ----------------------------------
            Factory A
            ----------------------------------
            pA: X111 + X112 + X113 + X114 ≤ 6
            pB: X211 + X212 + X213 + X114 ≤ 4
            pC: X311 + X312 + X313 + X314 ≤ 0
            pD: X411 + X412 + X413 + X414 ≤ 3
            ----------------------------------
            Factory B
            ----------------------------------
            pA: X121 + X122 + X123 + X124 ≤ 2
            pB: X221 + X222 + X223 + X224 ≤ 8
            pC: X321 + X322 + X323 + X324 ≤ 6
            pD: X421 + X422 + X423 + X424 = 0
            ----------------------------------
            Factory C
            ----------------------------------
            pA: X131 + X132 + X133 + X134 ≤ 7
            pB: X231 + X232 + X233 + X234 = 0
            pC: X331 + X332 + X333 + X334 = 0
            pA: X431 + X432 + X433 + X434 ≤ 10
            ----------------------------------
    
    - Factories must order enough materials to satisfy demand
        
        For each factory and product, they must order atleast required amount
        of materials for each product. The material requirement per product
        unit multiplied by the product quantity must be 
        
        ----------------------------------
        Factory A
        ----------------------------------
        Material     Product A                      Product B                      Product C                      Product D
        mA         ≥ 5(X111 + X112 + X113 + X114)                                                               + 3(X411 + X412 + X413 + X414)
        mB         ≥ 3(X111 + X112 + X113 + X114)                                + 7(X311 + X312 + X313 + X314) + 2(X411 + X412 + X413 + X414)
        mC         ≥                                2(X211 + X212 + X213 + X114) + 9(X311 + X312 + X313 + X314) + 4(X411 + X412 + X413 + X414)
        md         ≥                                5(X311 + X312 + X313 + X314)                                + 15(X411 + X412 + X413 + X414)
        ----------------------------------
        Factory B
        ----------------------------------
        mA         ≥ 5(X121 + X122 + X123 + X124)                                                               + 3(X421 + X422 + X423 + X424)
        mB         ≥ 3(X121 + X122 + X123 + X124)                                + 7(X321 + X322 + X323 + X324) + 2(X421 + X422 + X423 + X424)
        mC         ≥                                2(X221 + X222 + X223 + X224) + 9(X321 + X322 + X323 + X324) + 4(X421 + X422 + X423 + X424)
        md         ≥                                5(X221 + X222 + X223 + X224)                                + 15(X421 + X422 + X423 + X424)
        ----------------------------------
        Factory C
        ----------------------------------
        mA         ≥ 5(X131 + X132 + X133 + X134)                                                               + 3(X431 + X432 + X433 + X434)
        mB         ≥ 3(X131 + X132 + X133 + X134)                                + 7(X331 + X332 + X333 + X334) + 2(X431 + X432 + X433 + X434)
        mC         ≥                                2(X231 + X232 + X233 + X234) + 9(X331 + X332 + X333 + X334) + 4(X431 + X432 + X433 + X434)
        md         ≥                                5(X231 + X232 + X233 + X234)                                + 15(X431 + X432 + X433 + X434)
        ----------------------------------

    - Suppliers cannot sell more materials than they have in stock
    
        For all suppliers, and all materials, the total amount of materials
        ordered by each supplier cannot exceed the material stock limit of the 
        suppler.
        
        ----------------------------------
        Supplier A
        ----------------------------------
        mA = 5(X111 + X112 + X113 + X114) + 3(X411 + X412 + X413 + X414) +
             5(X121 + X122 + X123 + X124) + 3(X421 + X422 + X423 + X424) +
             5(X131 + X132 + X133 + X134) + 3(X431 + X432 + X433 + X434) ≤ 20
    
        mB = 3(X111 + X112 + X113 + X114) + 7(X311 + X312 + X313 + X314) + 2(X411 + X412 + X413 + X414) +
             3(X121 + X122 + X123 + X124) + 7(X321 + X322 + X323 + X324) + 2(X421 + X422 + X423 + X424) +
             3(X131 + X132 + X133 + X134) + 7(X331 + X332 + X333 + X334) + 2(X431 + X432 + X433 + X434) ≤ 20
        ----------------------------------
        Supplier B
        ----------------------------------
        mA = 5(X111 + X112 + X113 + X114) + 3(X411 + X412 + X413 + X414) +
             5(X121 + X122 + X123 + X124) + 3(X421 + X422 + X423 + X424) +
             5(X131 + X132 + X133 + X134) + 3(X431 + X432 + X433 + X434) ≤ 25
    
        mB = 3(X111 + X112 + X113 + X114) + 7(X311 + X312 + X313 + X314) + 2(X411 + X412 + X413 + X414) +
             3(X121 + X122 + X123 + X124) + 7(X321 + X322 + X323 + X324) + 2(X421 + X422 + X423 + X424) +
             3(X131 + X132 + X133 + X134) + 7(X331 + X332 + X333 + X334) + 2(X431 + X432 + X433 + X434) ≤ 50
        ----------------------------------
        Supplier C
        ----------------------------------
        mB = 3(X111 + X112 + X113 + X114) + 7(X311 + X312 + X313 + X314) + 2(X411 + X412 + X413 + X414) +
             3(X121 + X122 + X123 + X124) + 7(X321 + X322 + X323 + X324) + 2(X421 + X422 + X423 + X424) +
             3(X131 + X132 + X133 + X134) + 7(X331 + X332 + X333 + X334) + 2(X431 + X432 + X433 + X434) ≤ 10
    
        mC = 2(X211 + X212 + X213 + X114) + 9(X311 + X312 + X313 + X314) + 4(X411 + X412 + X413 + X414) +
             2(X221 + X222 + X223 + X224) + 9(X321 + X322 + X323 + X324) + 4(X421 + X422 + X423 + X424) +
             2(X231 + X232 + X233 + X234) + 9(X331 + X332 + X333 + X334) + 4(X431 + X432 + X433 + X434) ≤ 70
    
        mD = 5(X311 + X312 + X313 + X314) + 15(X411 + X412 + X413 + X414) +
             5(X221 + X222 + X223 + X224) + 15(X421 + X422 + X423 + X424) +
             5(X231 + X232 + X233 + X234) + 15(X431 + X432 + X433 + X434) ≤ 40
        ----------------------------------
        Supplier D
        ----------------------------------
        mC = 2(X211 + X212 + X213 + X114) + 9(X311 + X312 + X313 + X314) + 4(X411 + X412 + X413 + X414) +
             2(X221 + X222 + X223 + X224) + 9(X321 + X322 + X323 + X324) + 4(X421 + X422 + X423 + X424) +
             2(X231 + X232 + X233 + X234) + 9(X331 + X332 + X333 + X334) + 4(X431 + X432 + X433 + X434) ≤ 20
    
        mD = 5(X311 + X312 + X313 + X314) + 15(X411 + X412 + X413 + X414) +
             5(X221 + X222 + X223 + X224) + 15(X421 + X422 + X423 + X424) +
             5(X231 + X232 + X233 + X234) + 15(X431 + X432 + X433 + X434) ≤ 50
        ----------------------------------
        Supplier E
        ----------------------------------
        mA = 5(X111 + X112 + X113 + X114) + 3(X411 + X412 + X413 + X414) +
             5(X121 + X122 + X123 + X124) + 3(X421 + X422 + X423 + X424) +
             5(X131 + X132 + X133 + X134) + 3(X431 + X432 + X433 + X434) ≤ 30
    
        mD = 5(X311 + X312 + X313 + X314) + 15(X411 + X412 + X413 + X414) +
             5(X221 + X222 + X223 + X224) + 15(X421 + X422 + X423 + X424) +
             5(X231 + X232 + X233 + X234) + 15(X431 + X432 + X433 + X434) ≤ 40
    
    - Factories cannot supply negative products or order negative materials,
      Suppliers cannot sell negative materials
        
        0 ≤ X^ijk ≤ 1 
        0 ≤ Y^abc ≤ 1 

D - Identify the objective function for the Linear Programming model to 
minimise overall cost.

    - Minimise the cost of production and shipping of products to customers
        minimise((production + shipping) * X^ijk)
    - Minimise the cost of materials and shipping to factories
        minimise((material cost + material shipping) * Y^abc)
    
"""

"""
E - Implement and solve the Linear Programming model using the identified 
variables, constraints and objective function.
"""
# Instantiate the solver
solver = pywraplp.Solver('LPWrapper',
                         pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

# Define the model decision variables
# ===================================

# Instantiate LP decision variable holders
factory_deliveries = dict()
factory_orders = dict()

for factory in factories:
    # Vars for determinining how much to supply from each of the three
    # factories to each of the four customers - intVar is used to restrict the
    # solution to whole integer numbers
    for product in products:
        for customer in customers:
            v_name = '{f}-{p}-{c}'.format(f=factory, p=product, c=customer)
            factory_deliveries[(factory, product, customer)] = solver.IntVar(
                0, solver.infinity(), v_name)
    # Vars for determining how much materials to order from each supplier,
    # intVar is used again to restrict the solution to whole integer numbers
    for material in materials:
        for supplier in suppliers:
            v_name = '{f}-{m}-{s}'.format(f=factory, m=material, s=supplier)
            factory_orders[(factory, material, supplier)] = solver.IntVar(
                0, solver.infinity(), v_name)

# C - Define the model constraints
# ================================
# Each customer needs to have their demand met
# For each customer and product combination
for customer in customers:
    for product in products:
        # If the customer has a product demand in the customer_demand table
        if not pd.isna(customer_demand[customer][product]):
            # Add the demand constraint for that customer/product with lb and
            # ub set to the customer demand so it must be met
            c1 = solver.Constraint(float(customer_demand[customer][product]),
                                   float(customer_demand[customer][product]))
            # For each factory set the coefficient so the total products
            # delivered from each factory matches the customer demand
            for factory in factories:
                c1.SetCoefficient(
                    factory_deliveries[(factory, product, customer)], 1)

# Factories have limited production capacity
# For each factory/product combination
for factory in factories:
    for product in products:
        # If the factory produces the product
        if not pd.isna(production_capacity[factory][product]):
            # Set constraint with ub as the factory product capacity
            c2 = solver.Constraint(
                0, float(production_capacity[factory][product]))
            # For each customer set the constraint coefficient so their total
            # does not exceed factory capacity
            for customer in customers:
                c2.SetCoefficient(
                    factory_deliveries[(factory, product, customer)], 1)
        else:
            # Else the factory does not produce the product, set both lb and ub
            # to zero so it cannot produce any of the product
            c2 = solver.Constraint(0, 0)
            for customer in customers:
                c2.SetCoefficient(
                    factory_deliveries[(factory, product, customer)], 1)

# Suppliers cannot sell more materials than they have in stock
# For each supplier/material combination
for supplier in suppliers:
    for material in materials:
        # If the supplier stocks the material
        if not pd.isna(supplier_stock[material][supplier]):
            # Set the constraint so that the supplier stock limit cannot be
            # exceed, constraint ub is the supplier material stock quantity
            c3 = solver.Constraint(
                0, float(supplier_stock[material][supplier]))
            # For all factories ensure the sum of a given material from a given
            # supplier does not exceeed the supplier stock quantity
            for factory in factories:
                c3.SetCoefficient(
                    factory_orders[(factory, material, supplier)], 1)

# Factories must order enough materials to satisfy demand
for factory in factories:
    # For each factory and material order
    for material in materials:
        # list to hold quantity of current material ordered
        ordered_materials = list()
        # list to hold product material requirements
        ordered_product_requirements = list()
        # For each supplier get the quantity of material ordered from factory
        for supplier in suppliers:
            # If the supplier stocks the material
            if not pd.isna(supplier_stock[material][supplier]):
                # Add that factories ordered materials from the current
                # supplier to the list of factory ordered materials
                ordered_materials.append(factory_orders[(factory, material,
                                                         supplier)])
        # For each product
        for product in products:
            # If the factory produces the product and the product requires
            # the current material
            if not pd.isna(production_capacity[factory][product]) and not (
                    pd.isna(product_requirements[material][product])):
                # For each customer
                for customer in customers:
                    # If the customer ordered the product
                    if not pd.isna(customer_demand[customer][product]):
                        # Add the product material requirment multiplied by the
                        # the amount of products ordered by the factory for
                        # that customer
                        ordered_product_requirements.append(
                            float(product_requirements[material][product]) *
                            factory_deliveries[(factory, product, customer)])

        # Add constraint that the quantity of an ordered material must be
        # greater than or equal to the total amount of material the current
        # factory needs to manufacture all products that require the material
        solver.Add(sum(ordered_materials) >= sum(ordered_product_requirements))

# Cost functon
# ============
# Minimise combined cost of producting and shipping products to customers
objective = solver.Objective()
for factory in factories:
    for product in products:
        for customer in customers:
            if not pd.isna(customer_demand[customer][product]) and (
                    not pd.isna(production_capacity[factory][product])):
                objective.SetCoefficient(
                    factory_deliveries[(factory, product, customer)],
                    (float(production_cost[factory][product]) +
                     float(shipping_costs[customer][factory])))

# Minimise the combined cost of material quantity and shipping to factories
for factory in factories:
    for material in materials:
        for supplier in suppliers:
            if not pd.isna(supplier_stock[material][supplier]):
                objective.SetCoefficient(
                    factory_orders[(factory, material, supplier)],
                    (float(raw_material_cost[material][supplier]) +
                     float(raw_material_shipping[factory][supplier])))

objective.SetMinimization()
solver.Solve()

# Solution Post-Processing
# ========================
# Note: During the extraction of solution results for each section a lot of the
# functions were repeated, these were all combined into one block of functions
# to save on code

# Calculate order cost details for each factory for all sourced materials from
# all suppliers, each product will have their associated quantity, raw material
# cost, shipping cost and total tracked. Each factory will track the total
# cost of their supplier order, along with the total cost of materials from
# each supplier

# For each factory/supplier/material combination
factory_materials = dict()
for f in factories:
    factory_materials[f] = dict()
    factory_materials[f]['supplier_total'] = 0
    for s in suppliers:
        factory_materials[f][s] = dict()
        factory_materials[f][s]['material_total'] = 0
        for m in materials:
            # If the factory ordered that material from that supplier
            if factory_orders[(f, m, s)].solution_value() > 0:
                # Get the quantity ordered
                quantity = factory_orders[(f, m, s)].solution_value()
                # Calculate the cost of the raw materials
                cost = raw_material_cost[m][s] * quantity
                # Calculate the cost of shipping
                shipping = raw_material_shipping[f][s] * quantity
                # Calculate the total material cost
                total = cost + shipping
                # Add the material details to the dictionary
                factory_materials[f]['supplier_total'] += total
                factory_materials[f][s]['material_total'] += total
                factory_materials[f][s][m] = {
                    'quantity': quantity, 'raw_material_cost': cost,
                    'shipping_cost': shipping, 'total': total}

# Calculate factory manufacturing requirements for each factory and product.
# For each factory the total production cost across all products is tracked,
# for each product the quantity of products manufactured and the associated
# production cost is tracked (not including raw materials)

factory_products = dict()
# Get the total products manufactured by factories
for f in factories:
    factory_products[f] = dict()
    factory_products[f]['total_production_cost'] = 0
    for p in products:
        # Only track products that are produced by the factory
        if not pd.isna(production_capacity[f][p]):
            for c in customers:
                # Only track products that were ordered by the customer
                if not pd.isna(customer_demand[c][p]):
                    if factory_deliveries[(f, p, c)].solution_value() > 0:
                        # Get the quantity of products factory is delivering to
                        # customer
                        q = factory_deliveries[(f, p, c)].solution_value()
                        # Get the cost of producting a unit of the product
                        cost = production_cost[f][p]
                        # Get the cost of producing all products
                        p_cost = cost * q
                        # Add the cost to the running cost of manufacturing
                        # all products
                        factory_products[f]['total_production_cost'] += p_cost
                        # Track the quantity of the current product being
                        # produced and the associated cost
                        try:
                            factory_products[f][p]['quantity'] += q
                            factory_products[f][p]['production_cost'] += p_cost
                        except KeyError:
                            factory_products[f][p] = dict()
                            factory_products[f][p]['quantity'] = q
                            factory_products[f][p]['production_cost'] = p_cost

# Track the products factories are sending to customers, for each factory
# track the total cost of shipping all products, for each customer track the
# cost of shipping to that customer only, for each prodocut track the quantity
# of products going to that customer and the associated shipping cost.
factory_customers = dict()
# For each factory, customer, product combination
for f in factories:
    factory_customers[f] = dict()
    factory_customers[f]['shipping_total'] = 0
    for c in customers:
        factory_customers[f][c] = dict()
        factory_customers[f][c]['customer_shipping_total'] = 0
        for p in products:
            # If the factory is delivering the product to the customer
            if factory_deliveries[(f, p, c)].solution_value() > 0:
                factory_customers[f][c][p] = dict()
                # Get the amount quantity of products delivered from the
                # factory to the customer
                q = factory_deliveries[(f, p, c)].solution_value()
                # Get the cost of shipping a unit of the product
                s_unit_cost = shipping_costs[c][f]
                # Get the cost of shipping units of a product
                s_cost = s_unit_cost * q
                # Add results to dictionary
                factory_customers[f][c][p]['quantity'] = q
                factory_customers[f][c][p]['shipping'] = s_cost
                factory_customers[f][c]['customer_shipping_total'] += s_cost
                factory_customers[f]['shipping_total'] += s_cost

"""
F - Answer the question how much of each product each factory should order from
each supplier and how much this order will cost including shipping.
"""
print('#--------#\n| Part F |\n#--------#')
for f in factories:
    for s in suppliers:
        for m in materials:
            try:
                print('{f} | {s} | {m} | Quantity: {q} '
                      '| Total Cost: {t} ({mc} + {ms})'.format(
                        f=f, s=s, m=m,
                        q=factory_materials[f][s][m]['quantity'],
                        t=factory_materials[f][s][m]['total'],
                        mc=factory_materials[f][s][m]['raw_material_cost'],
                        ms=factory_materials[f][s][m]['shipping_cost']))
            except KeyError:
                pass

        if factory_materials[f][s]['material_total']:
            print('\t>> Total Order Cost: {t}\n'.format(
                s=s, t=factory_materials[f][s]['material_total']))

for f in factories:
    if factory_materials[f]['supplier_total']:
        print('- {f} Total Supplier Cost: {t}'.format(
            f=f, t=factory_materials[f]['supplier_total']))

"""
G - Answer the question how much of each product each factory should be
manufacturing and how much cost each factory is incurring for all products they 
manufacture.
"""
print('\n#--------#\n| Part G |\n#--------#')
for f in factories:
    for p in products:
        try:
            print('{f} | {p} | Quantity: {q}'.format(
                f=f, p=p, q=int(factory_products[f][p]['quantity'])))
        except KeyError:
            pass

    total_factory_cost = (factory_products[f]['total_production_cost'] +
                          factory_materials[f]['supplier_total'])

    print("{f}'s Total Cost Incurred: {t}\n".format(
        f=f, t=total_factory_cost))

"""
H - Answer the question which products and how many each factory is delivering 
to each customer and how much this will cost.

Note: Although Canvas discussion mentions no cost is asked for in H of pt1, 
there is a cost in the assignment spec, it is assumed that this is the cost of
each factory shipping products to each customer
"""
print('\n#--------#\n| Part H |\n#--------#')
for f in factories:
    for c in customers:
        for p in products:
            try:
                print('{f} | {c} | {p} | Quantity: {q} '
                      '| Shipping Total Cost: {t} '.format(
                        f=f, c=c, p=p,
                        q=int(factory_customers[f][c][p]['quantity']),
                        t=factory_customers[f][c][p]['shipping']))
            except KeyError:
                pass
    print("{f}'s Total Shipping Cost: {t}\n".format(
        f=f, t=factory_customers[f]['shipping_total']))

"""
I - Answer the question how much cost a unit of each product (including the
raw materials used for the manufacturing of the customer’s specific product,
the cost of manufacturing for the specific customer and all relevant shipping
costs) incurs for each individual customer.
"""
print('\n#--------#\n| Part I |\n#--------#')
print('**Note: Results are rounded to two decimal places**')
for f in factories:
    factory_material_stock = {}
    # Get the average cost for each material unit, this will be used to
    # determine the cost of each product in terms of the material requirements
    for m in materials:
        total_material = 0
        total_cost = 0
        for s in suppliers:
            try:
                quantity = factory_materials[f][s][m]['quantity']
                cost = factory_materials[f][s][m]['total']
                total_material += quantity
                total_cost += cost
            except KeyError:
                pass

        average_material_cost = total_cost / total_material
        factory_material_stock[m] = average_material_cost

    # For each product a customer buys, add the material requirement cost, the
    # production cost and the shipping
    for c in customers:
        for p in products:
            # Only calculate cost where factory is delivering a product to a
            # customer
            if factory_deliveries[(f, p, c)].solution_value() > 0:
                # Get the materials required for the current product
                product_materials = list()
                for m in materials:
                    if not pd.isna(product_requirements[m][p]):
                        product_materials.append(m)
                # Given the requirements of the product, calculate how much it
                # cost the factory to accquire the required materials
                material_cost = 0
                for m in product_materials:
                    # Get the raw material requirements for the current
                    # material
                    m_req = product_requirements[m][p]
                    # Add quantity of required materials multiplied by the
                    # average material unit cost price
                    material_cost += m_req * factory_material_stock[m]
                # Calculate the production and shipping cost of the current
                # product
                cost = production_cost[f][p]
                shipping = shipping_costs[c][f]
                # Calculate the total cost of the materials, production, and
                # shipping
                total_product_cost = (
                        material_cost + cost + shipping)

                print('{f} average cost of producing {p} for {c}: {t}'.format(
                    f=f, p=p, c=c, t=str(round(total_product_cost, 2))))
    print()
