using PrettyTables
using Plots
using Flux
using Statistics
using ProgressMeter

show_video = false
save_animation = true

## Numerical Solution type selection
    case = "central_diff"
    #case = "euler"
    println("-- Pendulum Solver - $case --")

    initial_phi = 1.0
    initial_vel = 0.0

## load Pendulum object
    include("Dynsys.jl")
    l = 1.0
    g = 10.0
    m = 1.0
    c = 5.0
    k = 300.0
    pendulum = Dynsys.Math_pendulum(l, g, m, c, k, initial_phi, initial_vel)


## load integrator and memory for the results
    time=2.0
    ts=2000.0 #timesteps
    dt=time/ts #timestep size
    Integ = Dynsys.Integrator(dt,ts) # delta_t & timesteps defined
    Integ.res_phi = zeros(Integ.timesteps)
    Integ.res_phi_dot = zeros(Integ.timesteps)
    t=zeros(Integ.timesteps)

## Initial value assignemnt to vector
    Integ.res_phi[1]=pendulum.phi
    Integ.res_phi_dot[1]=pendulum.phi_dot

# Calculate ϕ(t = -dt)
    if case == "central_diff"
        acceleration = -pendulum.k / pendulum.m * pendulum.phi
        pendulum.phi_prev = pendulum.phi + 0.5 * Integ.delta_t^2 * acceleration
    else
        pendulum.phi_prev = 0
    end

## Initial pendulum position display
    if show_video == true
        fig = Dynsys.create_fig(pendulum)
        Dynsys.plot_state(pendulum)
        display(fig)
    end

# Numerical Solution calculation
    println("Solving Numerically using $case")
    for i in 2:Integ.timesteps
        # integration step
        Dynsys.run_step(Integ,case,pendulum,i)
        
        # plot the state
        if show_video == true
            global fig = Dynsys.create_fig(pendulum)
            Dynsys.plot_state(pendulum)
            display(fig)
        end
        # save the step
        Integ.res_phi[i] = pendulum.phi
        if case == "euler"
            Integ.res_phi_dot[i] = pendulum.phi_dot        
        elseif case == "central_diff"
            if i>=3
                Integ.res_phi_dot[i-1]=pendulum.phi_dot
            end
            if i==Integ.timesteps
                Integ.res_phi_dot[Integ.timesteps]=(pendulum.phi-pendulum.phi_prev)/(Integ.delta_t)
                #Velocity calculation by Backward euler for the last step
            end        
        end
    t[i]=Integ.delta_t*(i-1)
    end
    println("Numerical Solution is complete")

# Adding Exact/ Analytical solution for simple pendulum

    w_sq= pendulum.k/pendulum.m
    r1 = (-pendulum.c + sqrt(complex(pendulum.c^2-4*w_sq)))/2
    r2 = (-pendulum.c - sqrt(complex(pendulum.c^2-4*w_sq)))/2
    println("Roots of characteristic eqn: r1 = $r1 ; r2 = $r2 ")
    α=real(r1)
    β=imag(r1)
    A=initial_phi
    B=(initial_vel-α*initial_phi)/β
    ϕ_exact = zeros(Integ.timesteps)

    for i in 1:Integ.timesteps
        ϕ_exact[i]=exp(α*t[i])*(A*cos(β*t[i])+B*sin(β*t[i]))
    end
    println("Analytical solution complete ")



# Plotting angular position against time
    p1=plot(t,Integ.res_phi,label="$case", legend=:topright, dpi=300)
    plot!(t,ϕ_exact, label="Analytical", legend=:topright,ls=:dash, dpi=300)
    xaxis!("Time in seconds")
    yaxis!("Angular displacement in rad")
    title!("Pendulum|Phi vs time|$case method")
    display(p1)
    #savefig("$case and Analytical_phi_vs_time_dt $dt.png")

# Plotting Angular velocity against time
    p2=plot(t,Integ.res_phi_dot,label="$case", legend=:topright, dpi=300)
    xaxis!("Time in seconds")
    yaxis!("Angular velocity in rad/s")
    title!("Pendulum|Phi_dot vs time|$case method|dt = $dt ")
    #display(p2)
    #savefig("$case _phi_dot_vs_time_dt $dt.png")

#### Neural Network Structure ####
# Scaling the dataset
    s = 10.0
    Integ.res_phi = Integ.res_phi.*s
# Create the training dataset
    num_train_timesteps=Int(0.35*Integ.timesteps)
    train_t=t[1:1:num_train_timesteps]'
    train_phi=Integ.res_phi[1:1:num_train_timesteps]'

# Create the validation dataset
    num_val_timesteps = num_train_timesteps + Int(0.10*Integ.timesteps)
    val_t = t[num_train_timesteps:1:num_val_timesteps]'
    val_phi = Integ.res_phi[num_train_timesteps:1:num_val_timesteps]'

# Create the testing dataset
    num_test_timesteps=num_val_timesteps + Int(0.55*Integ.timesteps)
    test_t=t[num_val_timesteps:1:num_test_timesteps]'
    test_phi=Integ.res_phi[num_val_timesteps:1:num_test_timesteps]'

# Create the input for training the NN
    training_input=Flux.DataLoader((train_t,train_phi),batchsize=40,shuffle=true)

# Define the NN and its hyperparameterst
    n_input = 1 # number of neuron(s) in input layer
    n_output = 1 # number of neuron(s) in output layer
    n_hidden =  30 # number of neurons in hidden layers
    act_fun = tanh # Hidden layer activation function

    pendulum_nn=Chain(Dense(n_input => n_hidden, act_fun),
    Dense(n_hidden => n_hidden, act_fun),
    Dense(n_hidden => n_hidden, act_fun),
    Dense(n_hidden => n_output))
    adapt_LR = true
    if adapt_LR == true
        η = 0.01 # Learning rate
    else
        η = 0.001
    end
    #β1 = 0.9 # Exponential decay of first momentum
    #β2 = 0.8 # Exponential decay of second momentum
    opt_state=Flux.setup(Adam(η),pendulum_nn)

# Define the loss function
    ϵ=1e-4
    r_points=200

    # Calculate residual positions
    t_center = collect(range(ϵ,time-ϵ,r_points))'
    t_left = t_center .- ϵ
    t_right = t_center .+ ϵ
    function Loss(pendulum_nn, x, y)
        mse = Flux.mse(pendulum_nn(x),y)
        
        # Calculate physics-based residual/loss 
        # Using Central Difference
        ϕ_center = pendulum_nn(t_center)
        ϕ_left = pendulum_nn(t_left)
        ϕ_right = pendulum_nn(t_right)

        ϕ_dot_center = (ϕ_right - ϕ_left)/(2*ϵ)   
        ϕ_ddot_center = (ϕ_right - 2*ϕ_center + ϕ_left)/ϵ^2
        r = ϕ_ddot_center+(pendulum.k/pendulum.m)*ϕ_center+pendulum.c*ϕ_dot_center
        res = sum(r.^2)/length(r)
        res_weight = 1e-4    #physics loss scaling factor
        total_loss = mse + res_weight*res
        #println("MSE loss: $mse")
        #println("Physics Loss: $res")
        return total_loss
    end

# Define maximum training loops, target accuracy for early stopping, load memory for loss vectors, animation object  creation
    max_itrs = 50000
    conv_itrs = max_itrs
    target_loss = 1e-6
    println("-- Training network for maximum of $max_itrs times --")
    training_loss = []
    validation_loss = []
    iteration_num=[]
    test_loss= []
    anim=Animation()

# Training loop

    @showprogress for epoch in 1:max_itrs
            Flux.train!(Loss, pendulum_nn, training_input, opt_state)
            push!(training_loss, Loss(pendulum_nn,train_t,train_phi)/s^2) #saving unscaled loss value for clear interpretation
            push!(iteration_num, epoch)
            push!(validation_loss, Loss(pendulum_nn,val_t,val_phi)/s^2) #saving unscaled loss value for clear interpretation
            push!(test_loss, Loss(pendulum_nn,test_t,test_phi)/s^2) #saving unscaled loss value for clear interpretation
        
            err = training_loss[epoch]
            if err <= target_loss
                println("Training Network converged in $epoch Iterations, Current Loss: $err ")
                global conv_itrs=epoch
                break
            end
            if epoch % 20 == 0 && save_animation == true || conv_itrs == epoch
                global p3=plot(t, Integ.res_phi ./s, label="Numerical Solution",yticks = -0.8:0.1:1.0,legend=:topright, lw=2, dpi=300)
                xlabel!("Time (t) in seconds")
                ylabel!("Angular displacement (ϕ) in radians")
                title!("Pendulum Solution Comparison")
                vline!([0.35*time], ls=:dot, lab="training limit")
                #scatter!(t_center', zeros(length(t_center)), label = "physics loss training locations", markersize = 2, markershape =:circle, markercolor =:blue, markeralpha = 0.5)
                plot!(t, pendulum_nn(t')' ./s,legend=:topright, label="NN Solution Itr no. $epoch",lw=2, ls=:dot, dpi=300)
                frame(anim)
                display(p3)
            end
        # Display info every 100 iterations
        if epoch % 100 == 0
            println("Training Network Itr: $epoch, Current Loss: $err")
        end
        if adapt_LR == true
            if epoch == 500
                Flux.adjust!(opt_state, 0.001)
            elseif epoch == 1000
                Flux.adjust!(opt_state, 0.0005)
            elseif epoch == 2000
                Flux.adjust!(opt_state, 0.00025)
            elseif epoch == 4000
                Flux.adjust!(opt_state, 0.000125)
            end
        end
    end
# Saving Animation
    gif(anim, "simple_harmonic_oscillator.gif", fps = 30)

# Plot Learning Curves
    p4=plot(iteration_num,training_loss,lw=:2,title="Learning curve",legend=:topright, label="Training Loss",dpi=300)
    #savefig("Training Loss vs Iterations.png")

    plot!(iteration_num,validation_loss,ls=:dot,lw=:2,label="Validation loss", dpi=300)
    #savefig("Validation Loss vs Iterations.png")
    plot!(iteration_num,test_loss,ls=:dot,lw=:2,label="Testing Loss",dpi=300)
    xaxis!("Training epoch")
    yaxis!("Loss (Log scale)", :log10)
    ylims!(1e-6,1e1)
    display(p4)
    savefig("Learning_curve.png")

# Print validation & test loss values
    final_val_loss =Loss(pendulum_nn,val_t,val_phi)/s^2 #loss value for unscaled magnitude of phi
    final_test_loss = Loss(pendulum_nn, test_t,test_phi)/s^2 #loss value for unscaled magnitude of phi
    println("Validation Loss = $final_val_loss, Testing loss = $final_test_loss")

# Network prediction output & Rescaling the data back to original order
    phi_nn = pendulum_nn(t') ./s
    Integ.res_phi = Integ.res_phi ./s
    
## Plot Trained Network Output and Compare
    #plot(t,ϕ_exact,label="Analytical Solution",lw=:2,legend=:topright,dpi=300)
    p5=plot(t, Integ.res_phi, label="Numerical Solution",yticks = -0.8:0.1:1.0,ylim = (-0.8,1.0),legend=:topright, lw=2, dpi=300)
    plot!(t, phi_nn', label="NN Solution", legend=:topright,lw=2, ls=:dot, dpi=300)
    vline!([0.35*time], ls=:dot, lab="training limit")
    vline!([0.45*time], ls=:dot, lab="validation limit")
    vline!([1.0*time], ls=:dot, lab="testing limit")
    xlabel!("Time (t) in seconds")
    ylabel!("Angular displacement (ϕ) in radians")
    title!("Pendulum Solution Comparison")
    display(p5)
    #scatter!(training_t',training_phi',label="training locations", markersize = 1, markershape =:circle, markeralpha = 0.5)
    #scatter!(t_center', zeros(length(t_center)), label = "physics loss training locations", markersize = 2, markershape =:circle, markercolor =:blue, markeralpha = 0.5)
    savefig("Pendulum_Solver_PINN.png")

# R-squared, or the coefficient of determination [statistical measure of how well a neural network model fits the data]
    ϕ_mean = mean(Integ.res_phi)
    #SST (Total Sum of Squares) is the sum of the squares of the differences between the actual values and the mean of the actual values
    SST = sum(abs2.(Integ.res_phi .- ϕ_mean))
    #SSE (Sum of Squared Errors) is the sum of the squares of the differences between the predicted values and the actual values
    SSE = sum(abs2.(Integ.res_phi - phi_nn'))
    R2 = 1-SSE/SST
    println("R2 value of the trained network = $R2")