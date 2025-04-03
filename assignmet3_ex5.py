import cvxpy as cp
import numpy as np


def compute_competitive_equilibrium(budgets, valuations, supply):
    """
    Compute a competitive equilibrium for a Fisher market using CVXPY, without
    specifying a particular solver. CVXPY will use whichever solver is installed
    by default.
    
    Parameters:
    -----------
    budgets     : array-like, shape (n,)
        The budgets b_i for each of the n players.
    valuations  : array-like, shape (n,m)
        valuations[i,j] = (linear) value of good j to player i.
    supply      : array-like, shape (m,)
        The total supply of each of the m goods (often all 1.0).
        
    Returns:
    --------
    x_val  : ndarray, shape (n,m)
        The allocation matrix: x_val[i,j] = amount of good j allocated to player i.
    prices : ndarray, shape (m,)
        The equilibrium price of each good.
    spent  : ndarray, shape (n,)
        The actual spending of each player.
    utils  : ndarray, shape (n,)
        The utility each player obtains from her allocated bundle.

    Like 'compute_competitive_equilibrium' but handles the case
    where a player's valuations are all zero by excluding them
    from the log-sum objective and forcing their allocation to be zero.
    """
    budgets = np.array(budgets, dtype=float)
    valuations = np.array(valuations, dtype=float)
    supply = np.array(supply, dtype=float)
    
    n = budgets.shape[0]   # number of players
    m = supply.shape[0]    # number of goods
    
    # x[i,j] = how much of good j is allocated to player i
    x = cp.Variable((n, m), nonneg=True)
    
    constraints = []
    objective = 0.0
    
    for i in range(n):
        ui = valuations[i, :] @ x[i, :]
        
        # Check if this player has strictly positive valuations
        # (or effectively zero if the sum is tiny).
        if np.allclose(valuations[i,:], 0):
            # If valuations are all zero:
            #   - They won't appear in the objective
            #   - Force x[i,:] = 0 because they don't gain anything
            constraints.append(x[i, :] == 0)
        else:
            # Normal case: enforce positivity and add to objective
            constraints.append(ui >= 1e-9)
            objective += budgets[i] * cp.log(ui)
    
    # Market-clearing constraints: sum_i x[i,j] <= supply[j]
    for j in range(m):
        constraints.append(cp.sum(x[:, j]) <= supply[j])
    
    prob = cp.Problem(cp.Maximize(objective), constraints)
    prob.solve()  # uses default solver
    
    # If the solver fails or is "infeasible", x.value might be None
    # so let's handle that gracefully.
    if x.value is None:
        # In edge cases, maybe the solver can't find a feasible solution
        return None, None, None, None
    
    x_val = x.value
    # Compute each player's utility = valuations[i,:] @ x_val[i,:]
    # If valuations=0, then utility=0 (which is fine).
    utils = np.array([valuations[i,:] @ x_val[i,:] for i in range(n)], dtype=float)
    
    # --- Extract dual variables => raw prices for last m constraints
    raw_prices_list = []
    for constr in prob.constraints[-m:]:
        raw_prices_list.append(constr.dual_value)
    raw_prices = np.array(raw_prices_list, dtype=float)
    
    # Avoid zero or negative prices
    raw_prices = np.maximum(raw_prices, 1e-12)
    
    # If the first player's valuations were all zero, then sum_for_player0=0 => alpha=∞
    # We handle that by checking if sum_for_player0 is near zero.
    sum_for_player0 = raw_prices @ x_val[0, :]
    if sum_for_player0 < 1e-12:
        # If the first player's spend is effectively zero, we pick another player to do the normalization
        # or just set alpha=1 so that raw_prices become "raw" with no normalizing.
        alpha = 1.0
    else:
        alpha = budgets[0] / sum_for_player0
    
    prices = alpha * raw_prices
    spent = np.array([prices @ x_val[i, :] for i in range(n)], dtype=float)
    
    return x_val, prices, spent, utils

def print_equilibrium_results(x_val, prices, spent, utils, budgets):
    """
    Print the results of the competitive equilibrium in a convenient format.
    
    Parameters:
    -----------
    x_val   : ndarray, shape (n,m)
        Allocation matrix: x_val[i,j] = allocation of good j to player i.
    prices  : ndarray, shape (m,)
        Equilibrium price for each good.
    spent   : ndarray, shape (n,)
        Actual spending of each player.
    utils   : ndarray, shape (n,)
        Utility of each player.
    budgets : array-like, shape (n,)
        Budgets for each player, for reference in printing.
    """
    n, m = x_val.shape
    
    print("=== Computed Allocation (x[i,j]) ===")
    for i in range(n):
        allocation_str = ", ".join(f"{x_val[i,j]:.4f}" for j in range(m))
        print(f"  Player {i+1}: [{allocation_str}]")
    print()
    
    print("=== Equilibrium Prices ===")
    for j, p in enumerate(prices):
        print(f"  Price(good_{j}) = {p:.4f}")
    print()
    
    print("=== Spending and Utilities ===")
    for i in range(n):
        print(f"  Player {i+1}:")
        print(f"    Budget = {budgets[i]:.2f}")
        print(f"    Spent  = {spent[i]:.2f}")
        print(f"    Utility= {utils[i]:.2f}")
        print()


############################ MORE TESTS ############################
"""
More examples:
Scenario 1: Two players, two goods, equal budgets.
Scenario 2: Three players, two goods, unequal budgets, and varying valuations.
Scenario 3: Three players, three goods, each player strongly favors a different good, with different budgets.
"""
def demo_test_scenarios():
    # Scenario 3: 2 players, 2 goods
    valuations3 = [
        [5.0, 1.0],  # Player 1
        [2, 6]   # Player 2
    ]
    endowment3 = [1, 1]  # 1 unit of each good
    budgets3 = [10, 10]
    print("\n=== SCENARIO 3: 2 Players, 2 Goods ===")
    x_val3, prices3, spent3, utils3 = compute_competitive_equilibrium(budgets3, valuations3, endowment3)
    print_equilibrium_results(x_val3, prices3, spent3, utils3, budgets3)
    
    # Scenario 4: 3 players, 2 goods, unequal budgets
    valuations4 = [
        [3, 2],  # Player 1
        [5, 1],  # Player 2
        [2, 5]   # Player 3
    ]
    endowment4 = [2, 1]  # More units of first good
    budgets4 = [10, 20, 5]
    print("\n=== SCENARIO 4: 3 Players, 2 Goods (Unequal Budgets) ===")

    x_val4, prices4, spent4, utils4 = compute_competitive_equilibrium(budgets4, valuations4, endowment4)
    print_equilibrium_results(x_val4, prices4, spent4, utils4, budgets4)

    # Scenario 5: 3 players, 3 goods, varied valuations
    valuations5 = [
        [10, 0, 2],   # Prefers good 1
        [1, 8, 5],    # Prefers good 2
        [3, 3, 9]     # Prefers good 3
    ]
    endowment5 = [2, 2, 2]  # 2 units each
    budgets5 = [15, 10, 5]  # Different budgets
    print("\n=== SCENARIO 5: 3 Players, 3 Goods (Varied Valuations) ===")
    x_val5, prices5, spent5, utils5 = compute_competitive_equilibrium(budgets5, valuations5, endowment5)
    print_equilibrium_results(x_val5, prices5, spent5, utils5, budgets5)

"""
More examples:

1) Complementary vs. Independent Valuations: Shows how the algorithm handles players with different valuation structures - one player values a specific good highly, another values multiple goods equally.

2) Extreme Value Differences: Tests how the market handles very strong preferences, where each player values one good 100x more than others.

3) Scarce Resources, Many Players: Shows competition dynamics when many players (5) compete for limited resources.

4) Different Valuation Scales: Tests whether the algorithm properly handles players who express their valuations on completely different scales (from 1-2 vs 100-200).

5) Abundant Resources: Tests the case where there's more than enough of each good for everyone - prices should be very low.

6) One Player Values Nothing: Tests an edge case where one player doesn't value any goods - the algorithm should handle this gracefully.
"""
def demo_interesting_examples():
    print("\n===== MORE INTERESTING EXAMPLES =====")
    
    # Example 6: Complementary goods
    # First player values good 1 and 2 together, second player values them independently
    print("\n=== Example 6: Complementary vs. Independent Valuations ===")
    valuations6 = [
        [1, 1, 10],  # Player 1 - values good 3 highly
        [5, 5, 1]    # Player 2 - values goods 1 and 2 equally
    ]
    endowment6 = [1, 1, 1]
    budgets6 = [50, 50]
    x_val6, prices6, spent6, utils6 = compute_competitive_equilibrium(budgets6, valuations6, endowment6)
    print_equilibrium_results(x_val6, prices6, spent6, utils6, budgets6)


    # Example 7: Extreme preferences
    # Players have extreme differences in how they value goods
    print("\n=== Example 7: Extreme Value Differences ===")
    valuations7 = [
        [100, 1, 1],    # Player 1 strongly prefers good 1
        [1, 100, 1],    # Player 2 strongly prefers good 2
        [1, 1, 100]     # Player 3 strongly prefers good 3
    ]
    endowment7 = [1, 1, 1]
    budgets7 =[33, 33, 34]
    x_val7, prices7, spent7, utils7 = compute_competitive_equilibrium(budgets7, valuations7, endowment7)
    print_equilibrium_results(x_val7, prices7, spent7, utils7, budgets7)
    
    # Example 8: Scarce resource with many players
    # Many players competing for limited resources
    print("\n=== Example 8: Scarce Resources, Many Players ===")
    valuations8 = [
        [5, 3, 1],    # Player 1
        [4, 4, 2],    # Player 2
        [3, 5, 2],    # Player 3
        [1, 2, 7],    # Player 4
        [2, 1, 7]     # Player 5
    ]
    endowment8 = [1, 1, 1]  # Limited supply
    budgets8 = [20, 20, 20, 20, 20]  # Equal budgets
    x_val8, prices8, spent8, utils8 = compute_competitive_equilibrium(budgets8, valuations8, endowment8)
    print_equilibrium_results(x_val8, prices8, spent8, utils8, budgets8)
    
    # Example 9: Unequal valuation scales
    # Players value goods on different scales (some more "passionate" than others)
    print("\n=== Example 9: Different Valuation Scales ===")
    valuations9 = [
        [2, 1],         # Player 1 - modest valuations
        [20, 10],       # Player 2 - higher scale valuations 
        [200, 100]      # Player 3 - very high scale valuations
    ]
    endowment9 = [1, 1]
    budgets9 = [100, 100, 100]  # Equal budgets
    x_val9, prices9, spent9, utils9 = compute_competitive_equilibrium(budgets9, valuations9, endowment9)
    print_equilibrium_results(x_val9, prices9, spent9, utils9, budgets9)

    # Example 10: Abundant resources
    # When there's more than enough for everyone
    print("\n=== Example 10: Abundant Resources ===")
    valuations10 = [
        [5, 3, 2],    # Player 1
        [2, 6, 4]     # Player 2
    ]
    endowment10 = [10, 10, 10]  # Plenty of each good
    budgets10 = [50, 50]
    x_val10, prices10, spent10, utils10 = compute_competitive_equilibrium(budgets10, valuations10, endowment10)
    print_equilibrium_results(x_val10, prices10, spent10, utils10, budgets10)
    
    # Example 11: One player values nothing
    # Edge case where one player doesn't value any goods
    print("\n=== Example 11: One Player Values Nothing ===")
    valuations11 = [
        [5, 3, 2],    # Player 1 - normal valuations
        [0, 0, 0]     # Player 2 - values nothing
    ]
    endowment11 = [1, 1, 1]
    budgets11 = [50, 50]
    x_val11, prices11, spent11, utils11 = compute_competitive_equilibrium(budgets11, valuations11, endowment11)
    print_equilibrium_results(x_val11, prices11, spent11, utils11, budgets11)


"""
Example 12 (Different Supply Amounts): 3 players compete for 3 goods, but with uneven resource availability (5, 2, 1 units). This tests how the algorithm allocates scarcities differently across goods.
Example 13 (6 Players, 9 Goods): A larger scenario to see how the algorithm scales with more players/goods. Each player has a different valuation profile with the same endowment.
"""
def demo_additional_examples():
    print("\n===== ADDITIONAL EXAMPLES =====")

    # Example 12: Uneven supply amounts
    # 3 players, 3 goods, each with a different supply level
    print("\n=== Example 12: Different Supply Amounts ===")
    valuations12 = [
        [4, 1, 2],   # Player 1
        [2, 5, 3],   # Player 2
        [1, 2, 6]    # Player 3
    ]
    endowment12 = [5, 2, 1]  # Different units available for each good
    budgets12 = [15, 15, 10]
    x_val12, prices12, spent12, utils12 = compute_competitive_equilibrium(budgets12, valuations12, endowment12)
    print_equilibrium_results(x_val12, prices12, spent12, utils12, budgets12)

    # Example 13: Larger case with 6 players (n=6) and 9 goods (m=9)
    print("\n=== Example 13: 6 Players, 9 Goods ===")
    valuations13 = [
        [1, 2, 0, 4, 1, 0, 2, 3, 5],   # Player 1
        [2, 2, 2, 2, 2, 2, 2, 2, 2],   # Player 2 - uniform valuations
        [3, 0, 1, 1, 1, 5, 2, 0, 1],   # Player 3
        [5, 3, 3, 0, 2, 4, 1, 1, 1],   # Player 4
        [0, 1, 1, 1, 6, 3, 2, 7, 1],   # Player 5
        [4, 2, 5, 2, 3, 1, 2, 1, 0]    # Player 6
    ]
    endowment13 = [1, 1, 1, 1, 1, 1, 1, 1, 1]  # 9 goods with varying supply
    budgets13 = [10, 10, 15, 20, 25, 30]       # Different budgets

    x_val13, prices13, spent13, utils13 = compute_competitive_equilibrium(budgets13, valuations13, endowment13)
    print_equilibrium_results(x_val13, prices13, spent13, utils13, budgets13)

if __name__ == "__main__":
    # Test with the example from the slides that was in the class:
    budgets_ex = [60.0, 40.0]
    valuations_ex = [
        [8.0,     4.0, 2.0],   # Player 1
        [2.0,    6.0,    5.0]    # Player 2
    ]
    supply_ex = [1.0, 1.0, 1.0]

    x_val, prices, spent, utils = compute_competitive_equilibrium(
        budgets_ex, valuations_ex, supply_ex
    )

    print("===== EXAMPLE 1 =====")
    print_equilibrium_results(x_val, prices, spent, utils, budgets_ex)


    # EXAMPLE 2: your new scenario (3 players, 2 goods).
    valuations2 = [
        [5.0, 5.0],
        [5.0, 5.0],
        [3.0, 6.0]
    ]
    budgets2 = [30.0, 30.0, 40.0]
    supply2 = [1.0, 1.0]  # each of the 2 goods has supply 1

    x_val2, prices2, spent2, utils2 = compute_competitive_equilibrium(budgets2, valuations2, supply2)
    
    print("===== EXAMPLE 2 =====")
    print_equilibrium_results(x_val2, prices2, spent2, utils2, budgets2)


    # Run the demo test scenarios
    print("\n=== Running additional test scenarios ===")
    demo_test_scenarios()

    demo_interesting_examples()

    demo_additional_examples()