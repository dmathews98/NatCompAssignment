gpcount = 0
def gp(**kwargs):
    global gpcount
    gpcount += 1
    printout(
        "ARGS: ",
        kwargs
    )

    printout(f"GP {gacount} Training for {kwargs['time_steps']} Generations with Population {kwargs['population_size']}")
    testgp = GP(
        population_size=kwargs['population_size'],
        mutation_rate=kwargs['mutation_rate'],
        crossover_rate=kwargs['crossover_rate'],
        flip_chance=kwargs['flip_chance'],
        whither_rate=kwargs['whither_rate'],
        growth_rate=kwargs['growth_rate']
    )
    best, over_time = testgp.run(
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
    nn = testgp.genotype_to_neural_net(best[1], datarr)
    nn.summary()
    printout(
        f"GP Best: ",
        testgp.evaluate(
            traindata,
            trainlab,
            testdata,
            testlab,
            epochs=kwargs['test_epochs'],
            batch=kwargs['batch'],
            seed=1721204
        ),
        f"\n With key: {best[1]}"
    )
    plt.plot(np.array(over_time))
    showout('plot_of_gp_accuracies_over_time.png')
    
    plot_data(testdata, testlab, nn, verbose=False, plotname=f'gp{gpcount}.png')

