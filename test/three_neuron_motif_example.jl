# Load the worked example's actual definitions, without its long simulation
# and plotting section or a dependency on the documentation environment.
module ThreeNeuronMotifExample
const source_path = joinpath(@__DIR__,"..","docs","literate",
    "06_three_neuron_inhibitory_motif.jl")
const definitions = first(split(read(source_path,String),
    "# ## Configure and run the comparison";limit=2))
include_string(@__MODULE__,definitions,source_path)
end

@testset "Three-neuron motif documentation example" begin
    E = ThreeNeuronMotifExample

    @testset "Inhibitory input control" begin
        input_i = zeros(2)
        @test E.set_inhibitory_input_for_target!(
            input_i,80.0,0.5,0.1,0.05,40.0) === nothing
        @test isapprox(input_i,[66.0,80.0];atol=1E-12,rtol=1E-12)
        E.set_inhibitory_input_for_target!(input_i,80.0,0.5,0.4,0.3,40.0)
        @test isapprox(input_i,[80.0,80.0];atol=1E-12,rtol=1E-12)
        E.set_inhibitory_input_for_target!(input_i,80.0,0.0,0.1,0.05,40.0)
        @test isapprox(input_i,[80.0,80.0];atol=1E-12,rtol=1E-12)
    end

    @testset "Coefficients, calibration, and analytical fixed points" begin
        for (B,γ) in ((1.0,123456.0),(0.0,10.0))
            rule = E.plasticity_for_target_rate(3E-6,B,γ)
            coefficients = E.get_motif_coefficients(rule)
            M = (B+1)/(4*(E.τ_syn+E.τ_plast)) +
                (B-1)/(4*(E.τ_syn+γ*E.τ_plast))
            @test isapprox(coefficients.M10,-M;atol=1E-12,rtol=1E-12)
            @test isapprox(coefficients.M01,M;atol=1E-12,rtol=1E-12)
            @test size(rule.zero_weight_mask) == (1,2)
            @test !any(rule.zero_weight_mask)
            @test rule.αpost == 0.0
            @test rule.weight_min == E.weight_min
            @test rule.weight_max == E.weight_max

            prediction = E.motif_predictions(rule)
            @test isapprox(prediction.r_E,25.0;atol=1E-12,rtol=1E-12)
            @test isapprox(prediction.w₁,0.171875;atol=1E-12,rtol=1E-12)
            @test isapprox(prediction.w₂,0.015625;atol=1E-12,rtol=1E-12)
            @test isapprox(prediction.r_E,
                E.h_E-(prediction.w₁+prediction.w₂)*E.r_I;
                atol=1E-12,rtol=1E-12)

            common_drift = rule.αpre*E.r_I +
                (rule.αpost+rule.B*E.r_I)*prediction.r_E
            drift₁ = common_drift + coefficients.M10*prediction.w₁*E.r_I +
                coefficients.M01*E.w_IE*prediction.r_E
            drift₂ = common_drift + coefficients.M10*prediction.w₂*E.r_I
            @test isapprox(drift₁,0.0;atol=1E-10,rtol=0.0)
            @test isapprox(drift₂,0.0;atol=1E-10,rtol=0.0)
            @test isapprox(prediction.τ_difference/prediction.τ_sum,
                prediction.ratio;atol=1E-12,rtol=1E-12)

            no_feedback = E.motif_predictions(rule,0.0)
            @test isapprox(no_feedback.w₁,no_feedback.w₂;
                atol=1E-12,rtol=1E-12)
            faster_rule = E.plasticity_for_target_rate(6E-6,B,γ)
            @test E.get_motif_coefficients(faster_rule) == coefficients
            @test isapprox(E.motif_predictions(faster_rule).τ_sum,
                prediction.τ_sum/2;atol=1E-12,rtol=1E-12)
        end
    end

    @testset "Matched sum timescales" begin
        reference_rate = E.plasticity_for_target_rate(1.0,1.0,123456.0)
        reference_covariance = E.plasticity_for_target_rate(1.0,0.0,10.0)
        η_covariance = 3E-6
        η_rate = η_covariance*E.motif_predictions(reference_covariance).c₂ /
            E.motif_predictions(reference_rate).c₂
        rate = E.motif_predictions(
            E.plasticity_for_target_rate(η_rate,1.0,123456.0))
        covariance = E.motif_predictions(
            E.plasticity_for_target_rate(η_covariance,0.0,10.0))
        @test isapprox(rate.τ_sum,covariance.τ_sum;atol=1E-10,rtol=1E-12)
        @test isapprox(rate.τ_sum/3600,0.2908093278463649;
            atol=1E-12,rtol=1E-12)
        @test rate.τ_difference > 10*covariance.τ_difference
    end

    @testset "Equal learning rates preserve fixed points" begin
        η = 3E-6
        rate = E.motif_predictions(E.plasticity_for_target_rate(η,1.0,123456.0))
        covariance = E.motif_predictions(E.plasticity_for_target_rate(η,0.0,10.0))
        for field in (:r_E,:w₁,:w₂)
            @test isapprox(getproperty(rate,field),getproperty(covariance,field);
                atol=1E-12,rtol=1E-12)
        end
        @test isapprox(rate.τ_sum,24.600246002460026;atol=1E-10,rtol=1E-12)
        @test isapprox(rate.τ_difference/60,11.11111111111111;
            atol=1E-12,rtol=1E-12)
        @test rate.τ_sum < covariance.τ_sum/40
        @test rate.τ_difference < covariance.τ_difference
        @test rate.τ_difference > 25*rate.τ_sum
    end

    @testset "Shared default recording interval" begin
        rule = E.plasticity_for_target_rate(3E-6,0.0,10.0)
        result = E.simulate_motif(rule;
            simulation_duration=2*E.recording_interval,seed=7)
        @test isapprox(result.rate_times,
            [E.recording_interval/2,1.5*E.recording_interval];
            atol=1E-12,rtol=1E-12)
        @test length(result.w₁) == 2
        @test all(diff(result.weight_times) .>= E.recording_interval)
    end

    @testset "Short seeded simulation and input callback" begin
        for (B,γ) in ((1.0,123456.0),(0.0,10.0))
            rule = E.plasticity_for_target_rate(3E-6,B,γ)
            result = E.simulate_motif(rule;simulation_duration=4.0,
                recording_interval=0.5,seed=42)
            repeat_result = E.simulate_motif(rule;simulation_duration=4.0,
                recording_interval=0.5,seed=42)
            @test !isempty(result.rate_times)
            @test !isempty(result.weight_times)
            @test length(result.rate_times) == length(result.rates_e) ==
                length(result.rates_i)
            @test length(result.weight_times) == length(result.w₁) ==
                length(result.w₂)
            @test size(result.final_weights) == (1,2)
            @test issorted(result.rate_times)
            @test issorted(result.weight_times)
            @test all(isfinite,result.rates_e)
            @test all(isfinite,result.rates_i)
            @test all(>=(0.0),result.rates_e)
            @test all(>=(0.0),result.rates_i)
            @test all(w -> E.weight_min <= w <= E.weight_max,result.w₁)
            @test all(w -> E.weight_min <= w <= E.weight_max,result.w₂)
            @test isapprox(result.w₁,repeat_result.w₁;atol=1E-12,rtol=1E-12)
            @test isapprox(result.w₂,repeat_result.w₂;atol=1E-12,rtol=1E-12)
            @test isapprox(result.rates_e,repeat_result.rates_e;
                atol=1E-12,rtol=1E-12)
            @test isapprox(result.rates_i,repeat_result.rates_i;
                atol=1E-12,rtol=1E-12)

            @test result.final_input_i[2] == E.r_I
            initial_input = E.r_I-E.w_IE*(E.h_E-2*E.weight_start*E.r_I)
            @test !isapprox(result.final_input_i[1],initial_input;
                atol=1E-8,rtol=0.0)

            # Updating on every spike eliminates the lag of the default
            # 0.2-second callback and lets us test its output directly.
            every_event = E.simulate_motif(rule;simulation_duration=4.0,
                recording_interval=0.5,input_update_interval=1E-12,seed=42)
            expected_input = zeros(2)
            E.set_inhibitory_input_for_target!(expected_input,E.r_I,E.w_IE,
                every_event.final_weights[1,1],every_event.final_weights[1,2],E.h_E)
            @test isapprox(every_event.final_input_i,expected_input;
                atol=1E-12,rtol=1E-12)
        end
    end
end
