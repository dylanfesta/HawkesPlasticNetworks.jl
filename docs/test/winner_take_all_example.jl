module WinnerTakeAllExample
const source_path = joinpath(@__DIR__,"..","literate",
    "07_winner_take_all_self_stabilizing_plasticity.jl")
const source = read(source_path,String)
const parameters = first(split(source,"# ### STDP kernel";limit=2))
const simulation = first(split(last(split(source,
    "# ### Build the feedforward network";limit=2)),
    "# ### Run the network and follow the ten weights";limit=2))
const inhibitory_parameters = first(split(last(split(source,
    "# ## Inhibitory winner-take-all";limit=2)),
    "# ### Inhibitory STDP kernel";limit=2))
const inhibitory_simulation = first(split(last(split(source,
    "# ### Build the inhibitory feedforward network";limit=2)),
    "# ### Run the inhibitory network";limit=2))
const inhibitory_prediction = first(split(last(split(source,
    "# ### Single-winner estimate with rate rectification";limit=2)),
    "# ### Build the inhibitory feedforward network";limit=2))
const combined_definitions = first(split(last(split(source,
    "# ## Both excitatory and inhibitory";limit=2)),
    "# ### Follow all six weights";limit=2))
const definitions = parameters*simulation*inhibitory_parameters*
    inhibitory_prediction*inhibitory_simulation*combined_definitions
include_string(@__MODULE__,definitions,source_path)
end

@testset "Winner-take-all documentation example" begin
    E = WinnerTakeAllExample

    @testset "Operating point and window" begin
        @test E.N == 10
        @test E.simulation_duration == 16*3600.0
        @test E.r_pre == E.r_target == 50.0
        @test E.h_post <= E.r_target/10
        @test isapprox(E.τ_minus,10*E.τ_plus;atol=1e-12,rtol=1e-12)
        @test isapprox(E.A_minus,4*E.A_plus;atol=1e-12,rtol=1e-12)
        @test isapprox(E.B,E.A_plus-E.A_minus;atol=1e-12,rtol=1e-12)
        @test isapprox(E.αpre+E.B*E.r_target,0.0;atol=1e-12)
        @test E.αpost == 0.0
        @test isapprox(E.shared_weight_target,0.09;atol=1e-12)
        @test isapprox(E.h_post+E.r_pre*E.N*E.shared_weight_target,E.r_target;
            atol=1e-12,rtol=1e-12)
        M = E.A_plus/(E.τ_syn+E.τ_plus)
        # Total excitation remains stable even with just one active synapse.
        @test abs(E.B)*E.r_pre > M
    end

    @testset "Single-winner motif fixed point" begin
        weights = fill(E.weight_start,1,1)
        rule = E.H.PlasticityAsymmetricSTDP(
            E.η,E.B,E.αpre,E.αpost,E.τ_plus,E.γ,weights)
        population_pre = E.H.PopulationExpKernelExcitatory(1,E.τ_syn;label="pre")
        @test isapprox(E.M,E.H.get_motif_coef_M10(rule,population_pre);
            atol=1e-12,rtol=1e-12)
        @test isapprox(E.winner_weight,27/26;atol=1e-12,rtol=1e-12)
        @test isapprox(E.winner_rate,740/13;atol=1e-12,rtol=1e-12)
        @test isapprox(E.αpre+E.B*E.winner_rate+E.M*E.winner_weight,0.0;
            atol=1e-12)
        @test E.B*E.r_pre+E.M < 0.0
        for (factor,expected_sign) in ((0.9,1),(1.1,-1))
            weight = factor*E.winner_weight
            rate = E.h_post+E.r_pre*weight
            drift = E.η*E.r_pre*(E.αpre+E.B*rate+E.M*weight)
            @test expected_sign*drift > 0.0
        end
        @test E.winner_rate > E.r_target
    end

    @testset "Seeded feedforward simulation" begin
        result = E.simulate_competition(
            simulation_duration=4.0,recording_interval=0.5,seed=42)
        repeated = E.simulate_competition(
            simulation_duration=4.0,recording_interval=0.5,seed=42)
        @test size(result.weights) == (length(result.weight_times),10)
        @test size(result.final_weights) == (1,10)
        @test result.weight_times[1] == 0.0
        @test all(result.weights[1,:] .== 0.001)
        @test length(result.weight_times) > 2
        @test all(diff(result.weight_times) .> 0.0)
        @test all(diff(result.weight_times[2:end]) .>= 0.5)
        @test all(result.weight_times .<= 4.0)
        @test all(isfinite,result.weights)
        @test all(result.weights .>= 0.0)
        @test all(isfinite,result.final_weights)
        @test all(result.final_weights .>= 0.0)
        @test !isapprox(result.final_weights,fill(E.weight_start,1,10);
            atol=1e-8,rtol=0.0)
        @test isapprox(result.weights,repeated.weights;atol=1e-12,rtol=1e-12)
        @test isapprox(result.weight_times,repeated.weight_times;
            atol=1e-12,rtol=1e-12)
        @test isapprox(result.rates_post,repeated.rates_post;
            atol=1e-12,rtol=1e-12)
        @test isapprox(result.rates_pre,repeated.rates_pre;
            atol=1e-12,rtol=1e-12)
        @test length(result.rate_times) == length(result.rates_post) ==
            length(result.rates_pre)
        @test all(isfinite,result.rates_post)
        @test all(isfinite,result.rates_pre)
        @test result.final_input_pre == fill(E.r_pre,10)
        @test result.final_input_post == [E.h_post]
        @test result.weight_min == 0.0
        @test result.weight_max == Inf
    end

    @testset "Inverted inhibitory window and competition" begin
        @test E.r_I == 25.0
        @test isapprox(E.η_inhibitory,5e-6;atol=1e-15,rtol=1e-12)
        @test E.h_post_inhibitory > E.r_target_inhibitory
        @test isapprox(E.αpre_inhibitory,-30.0;atol=1e-12)
        @test isapprox(E.shared_target_inhibitory,0.2;atol=1e-12)
        @test isapprox(E.A_left_inhibitory-E.A_right_inhibitory,E.B_inhibitory;
            atol=1e-12)
        @test E.A_left_inhibitory > E.A_right_inhibitory > 0.0
        rule = E.H.PlasticityAsymmetricSTDP(
            E.η_inhibitory,E.B_inhibitory,E.αpre_inhibitory,E.αpost_inhibitory,E.τ_plus_inhibitory,E.γ_inhibitory,
            fill(E.weight_start_inhibitory,1,E.N_inhibitory);inverted=true)
        population_i = E.H.PopulationExpKernelInhibitory(E.N_inhibitory,E.τ_syn_inhibitory;label="i")
        @test isapprox(E.M_inhibitory,E.H.get_motif_coef_M10(rule,population_i);
            atol=1e-12,rtol=1e-12)
        @test E.M_inhibitory > 0.0
        @test E.N_inhibitory*E.B_inhibitory*E.r_I > E.M_inhibitory
        @test isapprox(E.αpre_inhibitory+E.B_inhibitory*E.r_target_inhibitory,0.0;atol=1e-12)
        # Hold total inhibition fixed: differences now grow rather than decay.
        weights = fill(E.shared_target_inhibitory,E.N_inhibitory)
        weights[1] += 0.01
        weights[2] -= 0.01
        rate = E.h_post_inhibitory-E.r_I*sum(weights)
        drifts = E.r_I .* (E.αpre_inhibitory+E.B_inhibitory*rate .+
            E.M_inhibitory .* weights)
        @test drifts[1] > drifts[2]
        @test isapprox(drifts[1]-drifts[2],E.r_I*E.M_inhibitory*0.02;atol=1e-12)
    end

    @testset "Seeded inhibitory feedforward simulation" begin
        result = E.simulate_inhibitory_competition(
            simulation_duration=4.0,recording_interval=0.5,seed=42)
        repeated = E.simulate_inhibitory_competition(
            simulation_duration=4.0,recording_interval=0.5,seed=42)
        @test size(result.weights) == (length(result.weight_times),E.N_inhibitory)
        @test all(result.weights[1,:] .== E.weight_start_inhibitory)
        @test all(isfinite,result.weights)
        @test all(result.weights .>= 0.0)
        @test all(diff(result.weight_times) .> 0.0)
        @test all(result.weight_times .<= 4.0)
        @test all(isfinite,result.rates_post)
        @test all(isfinite,result.rates_pre)
        @test isapprox(result.weights,repeated.weights;atol=1e-12,rtol=1e-12)
        @test isapprox(result.rates_post,repeated.rates_post;atol=1e-12,rtol=1e-12)
        @test isapprox(result.rates_pre,repeated.rates_pre;atol=1e-12,rtol=1e-12)
        @test result.final_input_pre == fill(E.r_I,E.N_inhibitory)
        @test result.final_input_post == [E.h_post_inhibitory]
        @test !isapprox(result.final_weights,fill(E.weight_start_inhibitory,1,E.N_inhibitory);
            atol=1e-8,rtol=0.0)
        @test result.weight_min == 0.0
        @test result.weight_max == Inf
        frozen = E.simulate_inhibitory_competition(
            inhibitory_rate=20.0,learning_rate=0.0,
            simulation_duration=4.0,recording_interval=0.5,seed=42)
        @test frozen.final_input_pre == fill(20.0,E.N_inhibitory)
        @test frozen.weights == fill(E.weight_start_inhibitory,size(frozen.weights))
    end

    @testset "Clipped single-winner prediction" begin
        prediction = E.inhibitory_single_winner_prediction()
        @test isapprox(prediction.weight,3.027994297902742;atol=1e-10,rtol=1e-12)
        @test isapprox(prediction.rate,40.04795646268339;atol=1e-10,rtol=1e-12)
        @test 2.0 <= prediction.weight <= 5.0
        @test prediction.weight > E.h_post_inhibitory*E.τ_syn_inhibitory
        @test 0.0 < prediction.rate < E.r_target_inhibitory
        q = E.r_I*E.τ_syn_inhibitory
        a = E.τ_syn_inhibitory/E.τ_plus_inhibitory
        C = a*E.beta(a,q+2)
        z = E.h_post_inhibitory*E.τ_syn_inhibitory/prediction.weight
        @test isapprox(E.αpre_inhibitory+prediction.rate*
            (E.A_left_inhibitory-E.A_right_inhibitory*C*z^a),0.0;atol=1e-12)
        for factor in (0.9,1.1)
            perturbed_z = z/factor
            rate = prediction.rate*(perturbed_z/z)^q
            drift = E.αpre_inhibitory+rate*
                (E.A_left_inhibitory-E.A_right_inhibitory*C*perturbed_z^a)
            @test (1-factor)*drift > 0.0
        end

        # Check the analytic clipped rate against an independent fixed-weight network.
        E.Random.seed!(42)
        post = E.H.PopulationExpKernelExcitatory(1,E.τ_syn_inhibitory;label="post")
        pre = E.H.PopulationExpKernelInhibitory(1,E.τ_syn_inhibitory;label="pre")
        connection = E.H.ConnectionWithWeights(post,fill(prediction.weight,1,1),pre)
        connected_post = E.H.ConnectedPopulationExpKernel(
            post,[E.h_post_inhibitory],(connection,pre))
        connected_pre = E.H.ConnectedPopulationExpKernel(pre,[E.r_I])
        recorder = E.H.RecorderPopulationRate(post,3600.0;Δt=60.0)
        network = E.H.RecurrentNetworkExpKernel(
            (connected_post,connected_pre),(recorder,))
        E.H.reset!(network)
        time_now = 0.0
        while time_now < 3600.0
            time_now = E.H.dynamics_step!(time_now,network)
        end
        rates = E.H.get_content(recorder).rates
        @test isapprox(sum(rates)/length(rates),prediction.rate;atol=1.0,rtol=0.0)
    end
end

@testset "Winner-take-all examples have independent parameters" begin
    E = WinnerTakeAllExample
    imports = "using Random, HawkesPlasticNetworks; const H = HawkesPlasticNetworks;\n"
    for (name,definitions,simulate) in (
        (:inhibitory,E.inhibitory_parameters*E.inhibitory_prediction*E.inhibitory_simulation,
            :simulate_inhibitory_competition),
        (:combined,E.combined_definitions,:simulate_combined_competition))
        @testset "$name without earlier examples" begin
            isolated = Module(gensym(name))
            Base.include_string(isolated,imports*definitions,E.source_path)
            result = getfield(isolated,simulate)(
                simulation_duration=4.0,recording_interval=0.5,seed=42)
            reference = getfield(E,simulate)(
                simulation_duration=4.0,recording_interval=0.5,seed=42)
            for field in propertynames(reference)
                @test isapprox(getproperty(result,field),getproperty(reference,field);
                    atol=1e-12,rtol=1e-12)
            end
        end
    end
end
