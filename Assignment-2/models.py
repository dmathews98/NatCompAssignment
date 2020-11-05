from master.py import *
from nn.py import *
from ga.py import *
from pso.py import *

sgdcount = 0
def sgd(**kwargs):
    global sgdcount
    sgdcount += 1
    printout(
        "ARGS: ",
        kwargs
    )

    fitness, evaluate_func, dimensions, nn = prepare_neural_net(
        DataParameters.Q,
        traindata,
        trainlab,
        datarr,
        labarr
    )

    # This performance seems highly variable, usually appearing in the 0.8-0.9 range,
    # but can go much better if lucky.
    printout(f"Gradient Descent {sgdcount} Training for {kwargs['time_steps']} Epochs with {kwargs['population_size']} per Batch")

    scores = []
    best_net = None
    for i in range(int(kwargs['averaging'])):
        nn.set_weights(np.random.uniform(-1, 1, nn.get_weight_count()))
        nn.model.fit(
            x=traindata,
            y=trainlab,
            epochs=kwargs['time_steps'],
            verbose=0,
            batch_size=kwargs['population_size']
        )
        score = nn.evaluate(testdata, testlab)
        scores.append(score)
        if best_net is None:
            best_net = nn
        else:
            if score < min(scores):
                best_net = nn
        print(f"Model {i}: {score}")
    scores = np.array(scores)
    printout(f"Gradient Descent Over!  Avg Score: {scores.mean()} Deviation: {scores.std()} Best: {min(scores)} Worst: {max(scores)}")
    plt.plot(scores)
    showout(f'sgd_gradient_descent{sgdcount}_pop{kwargs["population_size"]}_time{kwargs["time_steps"]}_avg{kwargs["averaging"]}.png') #plt.show()
    plot_data(testdata, testlab, best_net, verbose=False, plotname="sgd_test_result{sgdcount}.png")

psocount = 0
def pso(**kwargs):
    global psocount
    psocount += 1
    printout(
        "ARGS: ",
        kwargs
    )

    fitness, evaluate_func, dimensions, nn = prepare_neural_net(
        DataParameters.Q,
        traindata,
        trainlab,
        datarr,
        labarr
    )
    
    printout(f"PSO {psocount} Training for {kwargs['time_steps']} Epochs with Population {kwargs['population_size']}")
    swarm, best = PSO(
        dim=dimensions,
        fitness_func=fitness,
        **kwargs,
        verbose=True
    ).run()
    nn.set_weights(best/DataParameters.SCALE)
    #print(nn.evaluate(testdata, testlab))
    printout(f"PSO Training Over!  Score: {nn.evaluate(testdata, testlab)}")
    
    plot_data(testdata, testlab, nn, verbose=False, plotname=f'pso{psocount}.png')

gacount = 0
def ga(**kwargs):
    global gacount
    gacount += 1
    printout(
        "ARGS: ",
        kwargs
    )

    printout(f"GA {gacount} Training for {kwargs['time_steps']} Generations with Population {kwargs['population_size']}")
    testga = GA(
        population_size=kwargs['population_size'],
        mutation_rate=kwargs['mutation_rate'],
        crossover_rate=kwargs['crossover_rate']
    )
    best, over_time = testga.run(
        traindata,
        trainlab,
        testdata,
        testlab,
        train_epochs=kwargs['train_epochs'],
        test_epochs=kwargs['test_epochs'],
        batch=kwargs['population_size'],
        generations=kwargs['time_steps']
    )
    #print(f"Best Model (Loss; {best[0]}): ")
    nn = testga.genotype_to_neural_net(best[1], datarr)
    nn.summary()
    printout(
        f"GA Best: ",
        testga.evaluate(
            traindata,
            trainlab,
            testdata,
            testlab,
            epochs=kwargs['test_epochs'],
            batch=kwargs['batch'],
            seed=1721204
        )
    )
    #print("GA Training Over!")
    #print("Plot of (Training) Accuracies over Time")
    plt.plot(np.array(over_time))
    #plt.show()
    showout('plot_of_ga_accuracies_over_time.png')
    
    plot_data(testdata, testlab, nn, verbose=False, plotname=f'ga{gacount}.png')



