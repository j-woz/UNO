    # Accurate (goes through dataset again)
    # train_steps = int(np.ceil(len(x_train) / generator_batch_size))
    # Create generators to calculate r2
    # r2_train_gen = data_generator(x_train, y_train, generator_batch_size, peek=False, verbose=False)
    # r2_val_gen = data_generator(x_val, y_val, generator_batch_size, peek=False, verbose=False)
    # # Initialize R2Callback with generators
    # r2_callback = R2Callback_accurate(
    #     model=model,
    #     r2_train_generator=r2_train_gen,
    #     r2_val_generator=r2_val_gen,
    #     train_steps=train_steps,
    #     validation_steps=validation_steps,
    #     train_ss_tot=train_ss_tot,
    #     val_ss_tot=val_ss_tot
    # )