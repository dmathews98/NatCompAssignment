from models.py import *

test = """
COMMENT; Here are some quick commands you can pass in as
COMMENT; a string for batching purposes!  I've put an example
COMMENT; here - note that it doesn't have any default args, so all
COMMENT; args I have here are **precisely** the args that we have available
COMMENT; to us!  So just change the numbers or create new lines, don't
COMMENT; change the parameter names or add/delete any.
COMMENT; For testing purposes, I've reduced all the time_steps requirements
COMMENT; to 10 or 1.  Multiply these by 100 for the actual timesteps I think
COMMENT; we should use, but run this script with the small timesteps first
COMMENT; to make sure everything is good on the server.
COMMENT;
COMMENT;
COMMENT; Also, add either 'I' or 'F' to front of argument, depending
COMMENT; on whether you want it as an int or a float
COMMENT;
COMMENT;
SGD; population_size:I5, time_steps:I10, averaging:I10
COMMENT;
COMMENT;
COMMENT; Instead of inertia (w), we calculate inertia in terms of alpha, by
COMMENT; traveling along the complexity bound.  We have a new parameter,
COMMENT; 'epsilon' which intuitively specifies how far into the complexity bound
COMMENT; we want to go.
COMMENT;
COMMENT;
PSO; total_a:F4.04, a1_percent:F0.5, epsilon:F-0.3, population_size:I5, time_steps:I10, a3:F0
COMMENT;
COMMENT;
COMMENT; train/test epochs concerns how long to train each organism in the population
COMMENT; train_epochs is used when training the GA, test_epochs is used only once at the end
COMMENT; after we're done evolving the population, to get a final result.
COMMENT;
COMMENT;
GA; population_size:I12, time_steps:I1, mutation_rate:F0.1, crossover_rate:F0.5, train_epochs:I100, test_epochs:I1000, batch:I5
"""

def run_script(script):
    line_by_line = script.strip().split('\n')
    for line in line_by_line:
        data = line.split(';')
        alg = data[0]
        commands = data[1].split(',')

        if alg == 'COMMENT':
            pass
        else:
            kwargs = dict({})
            for command in commands:
                kw, arg = command.strip().split(':')
                kw = kw.strip()
                head = arg.strip()[0]
                body = arg.strip()[1:]
                if head == 'I':
                    kwargs[kw] = int(body)
                elif head == 'F':
                    kwargs[kw] = float(body)
                else:
                    printout(f"WARNING: DIDN'T SPECIFY DATATYPE")
            if alg == 'SGD':
                sgd(**kwargs)
            elif alg == 'PSO':
                total_a = kwargs['total_a']
                a1_percent = kwargs['a1_percent']
                def get_w(a, c):
                    """
                    Calculates w given a if we skirt along the 'region of compexity'
                    AKA the value of w on the level curve (w-a)^2-2(w+a)+1=0, where we
                    can see that this is precisely when the eigenvalues coincide.
                    We want w \in [-1, 1], so we take the negative root of 4a-c in this equation;
                    taking the positive root yields much higher w
                    c is intuitively how far we are dipping into the region of complexity
                    By specifying how much more w should be than is needed to enter region
                    of complexity (so negative c will enter the region more!)
                    """
                    return c -np.sqrt(4*a) + a + 1
                inertia = get_w(total_a, -0.3)

                pso(
                    w=inertia,
                    a1=a1_percent * total_a,
                    a2=(1-a1_percent) * total_a,
                    a3=kwargs['a3'],
                    population_size=kwargs['population_size'],
                    time_steps=kwargs['time_steps'],
                    search_range=10,
                    constrainer=lambda x: x
                )
            elif alg == 'GA':
                ga(**kwargs)
            elif alg == 'GP':
                print("NOT IMPLEMENTED YET")
            else:
                printout(f"WARNING: Your process {alg} is not valid!")

run_script(test)