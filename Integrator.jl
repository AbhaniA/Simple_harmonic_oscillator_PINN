####################################
# Explicit Euler
#
# numeric integration file for the
# mathematical pendulum
#
# - explicit euler
# -
####################################


mutable struct Integrator
    delta_t::Float64
    timesteps::Int64
    Integrator(delta_t, timesteps) = new(delta_t, timesteps) #Constructor
    res_phi::Vector
    res_phi_dot::Vector
end

## run one integration time step
function run_step(int::Integrator, type, pendulum, step)
    if type == "euler"
        run_euler_step(int, pendulum, step)
    elseif type == "central_diff"
        run_central_diff_step(int, pendulum, step)
    else
        println("... integration type not understood ...")
    end
end

## euler integration time step
function run_euler_step(int::Integrator, pendulum::Math_pendulum, step) #sin(phi)~~phi |linearization
    #println("Running euler step")
    w_sq=pendulum.k/pendulum.m
    a=-w_sq*pendulum.phi-pendulum.c*pendulum.phi_dot
    
    pendulum.phi=pendulum.phi+int.delta_t*pendulum.phi_dot
    pendulum.phi_dot=pendulum.phi_dot+a*int.delta_t
end

## central difference time step
function run_central_diff_step(int::Integrator, pendulum::Math_pendulum, step) #sin(phi)~~phi |linearization
    #println("Running central difference step")
    w_sq=pendulum.k/pendulum.m
    phi_temp=pendulum.phi_prev #for the first iteration or 
        # 2nd element of vector phi. phi calculation by backward euler
    pendulum.phi_prev=pendulum.phi
    
    a = 1/int.delta_t^2 + pendulum.c/(2*int.delta_t)
    b = 2/int.delta_t^2 - w_sq
    c = pendulum.c/(2*int.delta_t)-1/int.delta_t^2
    pendulum.phi = (b*pendulum.phi+c*phi_temp)/a
    if step>=3
        pendulum.phi_dot=(pendulum.phi-phi_temp)/(2*int.delta_t)
    end
end 
