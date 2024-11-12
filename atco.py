import random
import copy
import pulp
from queue import Queue, Empty

class Agent:
    A=['same','other']
    a_count=len(A)
    #discounting
    alpha=0.1
    gamma = 1
    epsilon = 0.1
    def __init__(self,S):
        self.Q = {s: {a: 0 for a in Agent.A} for s in S}

    def update_Q(self,s,sn,a,r):
        if self.Q[sn]['same']>=self.Q[sn]['other']:
            max_q = self.Q[sn]['same']
        else:
            max_q = self.Q[sn]['other']
        self.Q[s][a]=self.Q[s][a]+Agent.alpha*(r+Agent.gamma*max_q-self.Q[s][a])

    def action(self, s):
        c=random.random()
        if self.Q[s]['same']>=self.Q[s]['other']:
            if c>=Agent.epsilon:
                return 'same'
            else:
                return 'other'
        else:
            if c>=Agent.epsilon:
                return 'other'
            else:
                return 'same'     


class ConstraintSystem:
    def __init__(self, no_of_agents, no_of_people):
        self.prev_action=[None for i in range(no_of_agents)]
        self.no_of_agents=no_of_agents
        self.no_of_people=no_of_people
        self.no_of_free=no_of_people-no_of_agents

    def checkActions(self, agent_actions):
        consys_actions=copy.deepcopy(agent_actions)
        penalise=[]
        count_extra = self.no_of_free - self.prev_action.count('same')
        for i in range(self.no_of_agents):
            if self.prev_action[i]!=agent_actions[i]:
                penalise.append(0)
            elif agent_actions[i]=='same': #will lead to overwork
                penalise.append(-100)
                consys_actions[i]='other'
            elif count_extra>0:
                count_extra -= 1
                penalise.append(0)
            else:
                penalise.append(-100) #will lead to no one being scheduled
                consys_actions[i]='same'
        self.prev_action=copy.deepcopy(consys_actions)
        return consys_actions, penalise
    

#define environment



class Environment:
    def __init__(self,S,no_of_positions,people):
        self.table={p: {s: None for s in S} for p in range(no_of_positions)}

        self.rest= Queue()
        #in final code, replace the below block with actual people names
        for person in people:
            self.rest.put(person)
        #initializing the first state
        for i in self.table:
            self.table[i][0] = self.rest.get()

    def schedule(self,actions,s,reward):
        sn=s+1
        removed=[]
        for p in self.table:
            if actions[p] == 'same':
                reward[p] -= 2
                self.table[p][sn]=self.table[p][s]
            else:
                if self.rest.empty():
                    reward[p]-=100
                    #print('oops')
                    return reward, True
                removed.append(self.table[p][s])
                self.table[p][sn]=self.rest.get()
                reward[p]-= 1
        for ppl in removed:
            self.rest.put(ppl)
        return reward, False
    

def scheduler(shift, required_positions, people):
    #World
    if shift ==1:
        no_of_time_blocks=8
    elif shift == 2:
        no_of_time_blocks=9
    elif shift == 3:
        no_of_time_blocks=15
    else:
        print("shift out of bounds, defaulting to 6hrs")
        no_of_time_blocks=8
    
    #states - time blocks
    S=[s for s in range(no_of_time_blocks)]
    terminal = no_of_time_blocks-1

    #initialize agents, 1 per position
    no_of_agents = required_positions
    agents = [Agent(S) for i in range(no_of_agents)]
    Agent.epsilon=0.1

    #train the agents
    constraints = ConstraintSystem(no_of_agents,len(people))
    no_of_episodes=100
    for e in range(no_of_episodes):
        env=Environment(S,no_of_agents,people)
        for s in S[:terminal]:
            agent_actions=[]
            for agent in agents:
                agent_actions.append(agent.action(s))
            consys_actions, reward = constraints.checkActions(agent_actions)
            reward,restart = env.schedule(consys_actions,s,reward)
            i=0
            for agent in agents:
                agent.update_Q(s,s+1,agent_actions[i],reward[i])
                i=i+1
            if restart:
                break

    #generate the schedule
    env=Environment(S,no_of_agents,people)
    Agent.epsilon=0
    for s in S[:terminal]:
        agent_actions=[]
        for agent in agents:
            agent_actions.append(agent.action(s))
        consys_actions, reward = constraints.checkActions(agent_actions)
        reward,restart = env.schedule(consys_actions,s,reward)
        i=0
        for agent in agents:
            agent.update_Q(s,s+1,agent_actions[i],reward[i])
            i=i+1
        if restart:
            print("failed to generate schedule")
            break
    
    return env.table
    


# A temporary atco generator. This is not part of the problem.
# As the atco employee list is expected to be given.
def generate_random_atcos(n,ratings):
    atcos = {}
    ratings_list = [r for r in ratings]
    max_r=len(ratings)
    
    for i in range(1, n+1):
        num_ratings = random.randint(1, max_r)  # Each atco has 1 or more ratings
        atco_ratings = random.sample(ratings_list, num_ratings)
        preferences = {}
        for rating in atco_ratings:
            preferences[rating] = random.randint(1, 20)  # Preference scores between 1 and 20
        atcos[f'Atco{i}'] = {
            'ratings': atco_ratings,
            'preferences': preferences
        }
    return atcos

def add_admin_atco(atcos, admin_atco_count,ratings):
    ratings_list = [r for r in ratings]
    admin_atco_name = f'GenShift_{admin_atco_count}'
    preferences = {rating: 10 for rating in ratings_list}  # Preference score of 10 for all ratings
    atcos[admin_atco_name] = {
        'ratings': ratings_list,
        'preferences': preferences
    }
    return atcos

# main function where the magic happens
def assign_atcos_to_ratings(ratings, atcos):
    admin_atco_count = 0  # Counter for admin atcos
    
    while True:
        # Create a list of all assignments and a dictionary of preferences
        assignments = []
        preferences = {}
        for atco, data in atcos.items():
            quals = data['ratings']
            prefs = data['preferences']
            for rating in quals:
                assignments.append((atco, rating))
                preferences[(atco, rating)] = prefs.get(rating, 0)  # Default to 0 if not specified

        # Create the problem variable to contain the problem data
        prob = pulp.LpProblem("Atco_Rating_Assignment_With_Preferences", pulp.LpMaximize)

        # Decision variables: x[(atco, rating)] = 1 if atco is assigned to rating, 0 otherwise
        x = pulp.LpVariable.dicts("assign", assignments, cat='Binary')

        # Objective Function: Maximize total preference score
        prob += pulp.lpSum([preferences[(atco, rating)] * x[(atco, rating)] for (atco, rating) in assignments]), "TotalPreferenceScore"

        # Constraints

        # 1. Each rating must be assigned the required number of atcos
        for rating in ratings:
            prob += (
                pulp.lpSum([x[(atco, rating)] for atco in atcos if (atco, rating) in assignments])
                == ratings[rating],
                f"RatingRequirement_{rating}"
            )

        # 2. Each atco must be assigned to exactly one rating
        for atco in atcos:
            prob += (
                pulp.lpSum([x[(atco, rating)] for rating in atcos[atco]['ratings']])
                <= 1,
                f"AtcoAssignment_{atco}"
            )

        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Check if a feasible solution was found
        if prob.status == 1:
            # Feasible solution found
            break
        else:
            # Infeasible solution; add an admin atco and try again
            admin_atco_count += 1
            atcos = add_admin_atco(atcos, admin_atco_count, ratings)

    # Output the results
    #print("Status:", pulp.LpStatus[prob.status])
    print()
    if prob.status == 1:
        #Assigned Atcos and Their Ratings
        total_pref_score = 0
        assigned_atcos = {}  # Dictionary to store assigned atcos per rating
        for atco, data in atcos.items():
            assigned = False
            for rating in data['ratings']:
                if x.get((atco, rating)) and x[(atco, rating)].varValue == 1:
                    pref_score = preferences[(atco, rating)]
                    total_pref_score += pref_score
                    if rating not in assigned_atcos:
                        assigned_atcos[rating] = []
                    assigned_atcos[rating].append(atco)
                    assigned = True
                    break  # Since the atco is assigned to one rating
            
            if not assigned:
                # If not assigned, find the rating with the highest preference
                best_rating = max(data['ratings'], key=lambda r: preferences[(atco, r)])
                print(f"{atco}: Not assigned to any rating, assigning to {best_rating} based on highest credit.")
                
                # Assign atco to the rating with the highest preference
                if best_rating not in assigned_atcos:
                    assigned_atcos[best_rating] = []
                assigned_atcos[best_rating].append(atco)
                total_pref_score += preferences[(atco, best_rating)]

        # Grouping atcos assigned to each rating into 5 groups
        shift = {}
        for rating, atcos_list in assigned_atcos.items():
            groups = split_list(atcos_list, 5)
            for i, group in enumerate(groups, 1):
                group_key = i
                if group_key not in shift:
                    shift[group_key] = {}
                shift[group_key][rating] = group


        #print(f"Total Preference Score: {total_pref_score}")
        print(f"Number of general shift atcos required: {admin_atco_count}\n")
    else:
        print("No feasible solution found.")
    return shift

# split the list for each grp into number of day-cycles
def split_list(lst, n):
    if len(lst) < n:
        return [lst[i:i+1] for i in range(len(lst))]
    else:
        k, m = divmod(len(lst), n)
        return [lst[i*(k+1) : i*(k+1)+(k+1)] if i < m else lst[i*k + m : (i+1)*k + m] for i in range(n)]

def display_shifts(shifts,shift_names):
    for group, ratings in shifts.items():
        print(f"{shift_names[group-1]}:")
        for rating, atcos_list in ratings.items():
            print(f"{rating}: {atcos_list}")
        print()

def display_shift_schedule(day_schedule,shift,ratings,timings):
    print(f"Schedule for {shift}")
    print("Time: ",end='\t\t')
    for time in timings:
        print(f"{time}",end='\t')
    print()
    for rating in ratings:
        table = day_schedule[shift][rating]
        for i in range(0, ratings[rating]):
            print(f"{rating} {i+1}:",end='\t')
            for j in table[i]:
                print(table[i][j], end='\t')
            print()
    print()

def shift_finder(shifts, id):
    for k, g_dict in shifts.items():
        for g, atco_ids in g_dict.items():
            if id in atco_ids:
                return k, g
    return None, None  # if id is not found



    
if __name__ == "__main__":
    # Code to execute if this script is run directly
    print("Welcome to the ATCO scheduling system.")

    ratings = {
    'Radar': 2,
    'Tower': 5,
    'Ground': 12,
    }
    print("The default ratings and requirements are as follows")
    for i in ratings:
        print(f"{i}:{ratings[i]}")

    choice=input("If you'd like to redefine the requirements enter 1. Else 2 : ")
    print()
    if choice == '1':
        ratings={}
        no_of_ratings=int(input("How many ratings do you need? : "))
        for i in range(no_of_ratings):
            r_name = input("Enter rating name: ")
            ratings[r_name] = input(f"Enter no of positions of rating {r_name}: ")
        print()
    
    # Calculate total atcos and initial required number of atcos per position
    no_of_atcos = copy.deepcopy(ratings)
    n = 0
    for r in ratings:
        x = ratings[r]
        no_of_atcos[r] = (x // 2 + (x - x // 2) * 2) * 5
        n += no_of_atcos[r]
    n+=int(input(f"Minimum number of ATCO officers required is {n}. Enter how may extra you would like: "))
    print()
    print(f"Randomly generating {n} ATCO officers with a random assignment of ratings and credit.")
    print()
    atcos = generate_random_atcos(n,ratings)
    choice=input("If you want to view the officers generated, press 1. Else 2 : ")
    if choice=='1':
        for i in atcos:
            print(f"{i} : {atcos[i]['preferences']}")
    print()

    #Generating shiftwise schedule
    print("Grouping ATCOs and generating shiftwise schedule ...")
    shifts = assign_atcos_to_ratings(no_of_atcos, atcos)
    shift_names=['Morning Shift','Afternoon Shift','Night Shift', 'Night Off', 'Full Off']
    timings = timings = {shift_names[0]: ['7.15', '8.00', '8.45', '9.30', '10.15', '11.00', '11.45', '12.30'],
                         shift_names[1]: ['13.15', '14.00', '14.45', '15.30', '16.15', '17.00', '17.45', '18.30', '19.15'],
                         shift_names[2]: ['20.00', '20.45', '21.30', '22.15', '23.00', '23.45', '0.30', '1.15', '2.00', '2.45', '3.30', '4.15', '5.00', '5.45', '6.30']
                         }
    
    #Generating daily roster
    print("Generating a complete daily roster ...")
    day_schedule = {}
    for i in range(1,4):
        shift=i
        day_schedule[shift_names[i-1]]= {rating: scheduler(shift,ratings[rating],shifts[shift][rating]) for rating in ratings}
    
    while True:
        print()
        choice=input("Do you want to see\n1. Complete day roster\n2. Shift superwiser view\n3. Single Shift Roster\n4. Atco view\n5. Exit\nEnter your choice: ")

        #Complete day roster
        if choice=='1':
            for shift in day_schedule:
                display_shift_schedule(day_schedule,shift,ratings,timings[shift])

        #Shift superwiser view
        elif choice=='2':
            display_shifts(shifts,shift_names)

        #Shift
        elif choice=='3':
            shift=int(input("Do you want to see schedule for \n1. Morning Shift\n2. Afternoon Shift\n3. Night Shift\nEnter your choice: "))
            shift=shift_names[shift-1]
            print()
            display_shift_schedule(day_schedule,shift,ratings,timings[shift])

        elif choice=='4':
            id='Atco'+input('Enter ATCO ID: ')
            assigned_shift,assigned_rating=shift_finder(shifts,id)
            if assigned_shift<=3:
                print(f'Assigned to {shift_names[assigned_shift-1]} - {assigned_rating}')
                assigned_schedule = {time: 'Break' for time in timings[shift_names[assigned_shift-1]]}
                table = day_schedule[shift_names[assigned_shift-1]][assigned_rating]
                i=0
                for work in assigned_schedule:
                    for p in table:
                        if table[p][i]==id:
                            assigned_schedule[work]=f'{assigned_rating} {p+1}'
                    i+=1
                for time, work in assigned_schedule.items():
                    print(f"{time} : {work}")
            else:
                print(shift_names[assigned_shift-1])

        else:
            print("Goodbye")
            break
    
