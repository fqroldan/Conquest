using QuantEcon, LinearAlgebra, Random

using PlotlyJS

mutable struct PLM
	κ::Float64          # Weight on current state (unemployment or inflation)
	γ::Vector{Float64}  # Weights on lagged states (unemployment + inflation + constant)

	Mu::Int64           # Lags on unemployment
	My::Int64           # Lags on inflation

	classic::Bool       # TRUE for predicting unemployment with the state, FALSE for predicting inflation

	g::Float64          # Gain parameter (only for constant-gain RLS)
	R::Matrix{Float64}  # Variance matrix (only for RLS)
	λ::Float64          # Adaptation parameter (only for simple adaptive expectations)
end

mutable struct ConquestEconomy
	δ::Float64          # Discount factor

	plm_g::PLM          # Government's perceived law of motion
	plm_p::PLM          # Private sector's perceived law of motion
	alm::PLM            # Actual (Nature's perceived) law of motion
		
	private_expectations::String

	σe::Float64         # Variance of cost-push shock
	σc::Float64         # Variance of inflation control shock
end

lags(plm::PLM) = length(plm.γ)

function sum_coeff(plm::PLM) # Finds the sum of coefficients on (current and) lagged inflation
	if plm.classic
		current_term = plm.κ
	else
		current_term = 0
	end
	return sum(plm.γ[plm.Mu+1:plm.Mu+plm.My]) + current_term
end

get_Ustar(alm::PLM) = alm.γ[end]
get_θ(alm::PLM) = -alm.κ
get_Ustar(ce::ConquestEconomy) = get_Ustar(ce.alm)
get_θ(ce::ConquestEconomy) = get_θ(ce.alm)

function make_ALM(plmg::PLM, plmp::PLM, Ustar::Float64, θ::Float64)
	Mu = max(plmg.Mu, plmp.Mu)
	My = max(plmg.My, plmp.My)
	γ = [zeros(Mu+My); Ustar]
	κ = -θ
	alm = PLM(κ, γ, Mu, My, true, 0.0, plmg.R, plmg.λ) # In the ALM, all the forecasting variables are irrelevant
	return alm
end

make_ALM(ce::ConquestEconomy, Ustar=get_Ustar(ce), θ=get_θ(ce)) = make_ALM(ce.plm_g, ce.plm_p, Ustar, θ)
function update_ALM!(ce::ConquestEconomy)
	ce.alm = make_ALM(ce)
	nothing
end

function ConquestEconomy(; 
	δ = 0.98,
	θ = 1.0,
	g = 0.05,
	λ = 0.5,  # λ is the weight on past observations for simple adaptive scheme
	σe = 0.5,
	σc = 0.5,
	Ustar = 5.0,
	classic = true,
	private_expectations = "rational",

	Mu = 1,
	My = 1,
	Mu_p = 2,
	My_p = 2
	)

	# Set arbitrary coefficients on past inflation and past unemployment
	γU = zeros(Mu)
	γy = zeros(My)
	γ = [γU; γy; 1]
	# For the induction hypothesis (weights on y's sum up to zero)
	κ = -sum(γy)
	R = zeros(Mu+My+1, Mu+My+1)

	plm_g = PLM(κ, γ, Mu, My, classic, g, R, λ)

	if private_expectations == "RLS"
		priv_gain = 0.01
	else
		Mu_p, My_p, priv_gain = Mu, My, g
	end
	Rp = zeros(Mu_p+My_p+1, Mu_p+My_p+1)
	classic = true # Always use the classical scheme for the private sector
	plm_p = PLM(1.0, [zeros(Mu_p+My_p);1], Mu_p, My_p, classic, priv_gain, Rp, λ)

	# True process
	alm = make_ALM(plm_g, plm_p, Ustar, θ)

	return ConquestEconomy(δ, plm_g, plm_p, alm, private_expectations, σe, σc)
end

function set_exp!(ce::ConquestEconomy, s::String=""; Mu=1, My=1, Mu_p=2, My_p=2)
	if s != "" && s != "rational" && s != "adaptive" && s != "RLS"
		println("WARNING: only 'rational', 'adaptive', and 'RLS' expectations allowed.")
	else
		if s == ""
			s = ce.private_expectations
		end    
		ce2 = ConquestEconomy(private_expectations = s, Mu=Mu, My=My, Mu_p=Mu_p, My_p=My_p)
		ce.plm_p = ce2.plm_p
		ce.plm_g = ce2.plm_g
		ce.private_expectations = s
		update_ALM!(ce)
	end
	nothing
end

function set_exp!(ce::ConquestEconomy, n::Int64=0; Mu=1, My=1, Mu_p=2, My_p=2)
	if n == 0
		set_exp!(ce, "", Mu=Mu, My=My, Mu_p=Mu_p, My_p=My_p)
	else
	sv = ["rational"; "adaptive"; "RLS"]
	set_exp!(ce, sv[n], Mu=Mu, My=My, Mu_p=Mu_p, My_p=My_p)
	end
	nothing
end

function make_LQ(plm::PLM, δ, σc, σe)
	# Law of motion x_{t+1} = A x_t + B u_t + C w_{t+1}
	# u = mean inflation, y = u + σc ϵ

	if plm.classic
		κ, γ = plm.κ, plm.γ
	else
		# Invert the Phillips curve in the Keynesian formulation
		κ, γ = 1/plm.κ, -plm.γ / plm.κ
	end

	k = lags(plm)

	# Law of motion
	A = zeros(k, k)
	A[1,:] = γ[:]
	for jj in 1:plm.Mu-1
		A[jj+1, jj] = 1
	end
	for jj in 1:plm.My-1
		A[plm.Mu+jj+1, plm.Mu+jj] = 1
	end
	A[end,end] = 1

	B = zeros(k,1)
	B[1,1] = κ
	B[plm.Mu+1,1] = 1
	
	# Shocks: unemployment is affected directly by the cost-push shock and indirectly by the inflation control shock (κ-to-1), inflation affected directly by the control shock
	C = zeros(k,2)
	C[1,:] = [sqrt(σe) κ*sqrt(σc)]
	C[plm.Mu+1, 2] = sqrt(σc)

	# Set up objective function
	# r = x'Rx + u'Qu + 2 u'Nx

	Q = [1+κ^2 for ji in 1:1, jj in 1:1]
	R = γ * γ'
	N = [κ * γ[jj] for jy in 1:1, jj in 1:k]

	lq = QuantEcon.LQ(Q, R, A, B, C, N, bet = δ)
	return lq
end

function solve_Phelps!(lq::QuantEcon.LQ)
	try
		# Use QuantEcon's LQ solver
		P, F, d = stationary_values!(lq)

		# Bind states
		lq.P, lq.F, lq.d = P, F, d
	
	catch
		println("Default solver exited. Increasing maxiter.")
		# Increase number of iterations by hand
		Q, R, A, B, N, C = lq.Q, lq.R, lq.A, lq.B, lq.N, lq.C

		# solve Riccati equation, obtain P
		A0, B0 = sqrt(lq.bet) * A, sqrt(lq.bet) * B
		P = solve_discrete_riccati(A0, B0, R, Q, N, max_it=100)

		# Compute F
		s1 = Q .+ lq.bet * (B' * P * B)
		s2 = lq.bet * (B' * P * A) .+ N
		F = s1 \ s2

		# Compute d
		d = lq.bet * tr(P * C * C') / (1 - lq.bet)

		# Bind states
		lq.P, lq.F, lq.d = P, F, d
	end

	nothing
end

function expand_index(plm::PLM, alm::PLM, jj::Int64)
	new_j = jj
	if jj > plm.Mu && jj <= plm.Mu+plm.My
		new_j = alm.Mu - plm.Mu + jj
	elseif jj == plm.Mu + plm.My + 1
		new_j = alm.Mu + alm.My + 1
	end
	new_j
end

function expand_matrix(plm::PLM, alm::PLM, M::Matrix)
	A = zeros(expand_index(plm, alm, size(M,1)), expand_index(plm, alm, size(M,2)))

	for ji in 1:size(M,1)
		for jj in 1:size(M,2)
			A[expand_index(plm, alm, ji), expand_index(plm, alm, jj)] = M[ji,jj]
		end
	end
	A
end

function simul_PLM(ce::ConquestEconomy, plm::PLM)
	lq = make_LQ(plm, ce.δ, ce.σc, ce.σe)

	xpath, upath, wpath = compute_sequence(lq, ones(lags(plm)))
end

function update_PC_OLS!(plm::PLM, alm::PLM, Ut, yt, Xt)

	if plm.classic
		predicted_var = Ut
		predictor_var = yt
	else
		predicted_var = yt
		predictor_var = Ut
	end

	# States has lags on unemployment, lags on inflation, and the constant
	States = [Xt[1:plm.Mu]; Xt[alm.Mu+1:alm.Mu+plm.My]; Xt[end]]

	# OLS formula U_t = κ y_t + γ'X_t + ϵ_t
	X = [predictor_var States']'

	# Implement OLS inv(x'x)x'y with x \ y
	new_γ = X' \ predicted_var

	# Bind coefficients
	plm.κ = new_γ[1]
	plm.γ = new_γ[2:end]

	new_γ
end

function update_PC_RLS!(plm::PLM, alm::PLM, Ut, yt, Xt)

	if plm.classic
		predicted_var = Ut
		predictor_var = yt
	else
		predicted_var = yt
		predictor_var = Ut
	end

	# States has lags on unemployment, lags on inflation, and the constant
	States = [Xt[1:plm.Mu]; Xt[alm.Mu+1:alm.Mu+plm.My]; Xt[end]]

	X = [predictor_var States']'

	old_γ = [plm.κ; plm.γ[1:end]]

	# Use RLS updating formulas
	new_R = plm.R + plm.g * (X*X' - plm.R)
	new_γ = old_γ + plm.g * (new_R \ (X*(predicted_var - old_γ'*X)))

	# Bind new coefficients
	plm.κ = new_γ[1]
	plm.γ = new_γ[2:end]
	plm.R = new_R

	new_γ
end


function simul_reest(ce::ConquestEconomy, T=50; OLS::Bool=false)
	Random.seed!(1)
	
	# Unpack and prepare
	plm_g, plm_p, alm = ce.plm_g, ce.plm_p, ce.alm
	expectations = ce.private_expectations
	alm_lq = make_LQ(alm, ce.δ, ce.σc, ce.σe)
	solve_Phelps!(alm_lq)

	# Initial state
	xv, uv, wv = simul_PLM(ce, plm_p)
	initial_T = 50
	xv, uv, wv = xv[:,end-initial_T:end], uv[:,end-initial_T+1:end], wv[:,end-initial_T:end]
	xnew = xv[:,end]

	# Ensure the constant is a constant (mostly redundant)
	xv[end,end] = 1

	# Initial variance matrix
	yvec = xv[alm.Mu+1, 2:end]
	States_vec = vcat(xv[1:plm_g.Mu,1:end-1], xv[alm.Mu+1:alm.Mu+plm_g.My, 1:end-1], xv[end:end, 1:end-1])
	X = [yvec States_vec']'
	plm_g.γ = zeros(size(plm_g.γ))
	plm_g.κ = 0.0
	plm_g.R = X*X'

	# Initialize private sector
	States_vec = vcat(xv[1:plm_p.Mu,1:end-1], xv[alm.Mu+1:alm.Mu+plm_p.My, 1:end-1], xv[end:end, 1:end-1])
	X = [yvec States_vec']'
	plm_p.R = X*X'
	x_priv = 0.0
	
	# Initialize vectors to store results
	svec = ones(size(xv,2)-1) * sum_coeff(plm_g)
	κvec = ones(size(xv,2)-1) * plm_g.κ
	γvec = [plm_g.γ[jj] for jy in 1:1, jj in 1:lags(plm_g)]
	πv = vcat(yvec, zeros(T))
	Uv = vcat(xv[1, 2:end], zeros(T))

	for jj in initial_T+1:initial_T+T
		if OLS
			plm_g.g = min(plm_g.g, 1/jj)
		end

		Xt = xnew

		# Gov't chooses its action
		lq = make_LQ(plm_g, ce.δ, ce.σc, ce.σe)
		solve_Phelps!(lq)

		# Private sector sets expectations
		Fp = zeros(1, lags(plm_p))
		if expectations == "adaptive" || expectations == "RLS"
			Fp[1,end] = x_priv # Add expected inflation in the constant term
		elseif expectations == "rational"
			Fp = expand_matrix(plm_g, plm_p, lq.F)
		end

		# Nature determines the natural rate given current private expectations
		θ = get_θ(alm)
		Ustar = get_Ustar(alm) + (θ * expand_matrix(plm_p, alm, -Fp) * Xt)[1]
		alm_lq.A[1,end] = Ustar

		# Gov't implements choice of action
		
		unew = expand_matrix(plm_g, alm, -lq.F) * Xt

		# Nature sets the state with the real model
		shocks = randn(2)
		wnew = alm_lq.C * (shocks .* [sqrt(ce.σe), sqrt(ce.σc)])
		xnew = alm_lq.A*xnew + alm_lq.B*unew + wnew

		# Save output and unemployment
		Ut = xnew[1]
		yt = xnew[alm.Mu+1]

		# Private sector updates expectations for tomorrow
		if expectations == "adaptive"
			x_priv = x_priv + (1-plm_p.λ) * ((unew[1] + shocks[2]) - x_priv)
		end
		if expectations == "RLS"
			update_PC_RLS!(plm_p, alm, Ut, yt, xnew)
			lq_p = make_LQ(plm_g, ce.δ, ce.σc, ce.σe)
			solve_Phelps!(lq_p)

			x_priv = (expand_matrix(plm_g, alm, -lq_p.F) * Xt)[1]
		end
		
		# Government updates
		new_γ = update_PC_RLS!(plm_g, alm, Ut, yt, Xt)
		κvec = [κvec; plm_g.κ]
		svec = [svec; sum_coeff(plm_g)]
		γvec = [γvec; plm_g.γ']

		Uv[jj] = Ut
		πv[jj] = yt
	end

	print("Finished simulation with $expectations expectations")

	pU = plot(Uv, name="Unemployment", showlegend=false, Layout(title="Unemployment", yaxis_zeroline=false))
	pπ = plot(πv, name="Inflation", showlegend=false, Layout(title="Inflation", yaxis_zeroline=false))
	pκ = plot(κvec, name="PC slope coefficient", showlegend=false, Layout(title="PC slope coefficient", yaxis_zeroline=false))
	ps = plot(svec, name="Sum of lagged coefficients", showlegend=false, Layout(title="Sum of lagged coefficients", yaxis_zeroline=false))

	pl = [pU pπ; pκ ps]
	return pl
end