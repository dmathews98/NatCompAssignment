from models import *
import sys

def run_script(script):
    f = open(script, 'r')
    line_by_line = f.read().strip().split('\n')
    f.close()
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
                elif head == 'B':
                    kwargs[kw] = (body == 'True')
                else:
                    printout(f"WARNING: DIDN'T SPECIFY DATATYPE")
            if alg == 'SGD':
                sgd(**kwargs)
            elif alg == 'PSO':
                printout(f"Raw kwargs: {kwargs}")
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
                inertia = get_w(total_a, kwargs['epsilon'])

                pso(
                    w=inertia,
                    a1=a1_percent * total_a,
                    a2=(1-a1_percent) * total_a,
                    a3=kwargs['a3'],
                    population_size=kwargs['population_size'],
                    time_steps=kwargs['time_steps'],
                    search_range=DataParameters.SCALE,
                    constrainer=lambda x: x
                )
            elif alg == 'GA':
                ga(**kwargs)
            elif alg == 'GP':
                gp(**kwargs)
            elif alg == 'DATA_PARAMETER':
                for key, value in kwargs.items():
                    printout(f'Changed Parameters: {key}->{value}')
                    setattr(DataParameters, key, value)
                    # Now we remake all the data!
                    make_data.make_data()
            else:
                printout(f"WARNING: Your process {alg} is not valid!")

if __name__ == "__main__":
    run_script(sys.argv[1])
