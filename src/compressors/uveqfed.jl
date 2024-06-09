using Statistics
using LinearAlgebra

function nearestneighbour(P, X)
    # %NEARESTNEIGHBOUR    find nearest neighbours
    # %   IDX = NEARESTNEIGHBOUR(P, X) finds the nearest neighbour by Euclidean
    # %   distance to each point in P from X. P and X are both matrices with the
    # %   same number of rows, and points are the columns of the matrices. Output
    # %   is a vector of indices into X such that X(:, IDX) are the nearest
    # %   neighbours to P
    # %
    # # %     % Find the nearest neighbours to each point in p
    # %     p = rand(2, 5);
    # %     x = rand(2, 20);
    # %     idx = nearestneighbour(p, x)
    # %
    # % Default parameters
    no_neighbors = 1    ; # Finds one
    
    # % If it didn't use Delaunay triangulation, find the neighbours directly by
    # % finding minimum distances
    idx = zeros(Int32, no_neighbors, size(P, 2));
    # % Loop through the set of points P, finding the neighbours
    Y = zeros(size(X));
    for iPoint = 1:size(P, 2)
        x = P[:, iPoint];
        # % This is the faster than using repmat based techniques such as
        # % Y = X - repmat(x, 1, size(X, 2))
        for i = 1:size(Y, 1)
            Y[i, :] = X[i, :] .- x[i];
        end
        # % Find the closest points
        dSq = sum(abs.(Y).^2, dims=1);
        iRad = 1:size(dSq, 2);
        iSorted = iRad[findmin(dSq[iRad])[2]]; # minn(dSq[iRad], no_neighbors)
        # % Remove any bad ones
        idx[1:length(iSorted), iPoint] .= iSorted';
    end
    # if isvector(idx)
        # idx = idx[:]';
    # end
    return idx
end # % nearestneighbour 

function m_fGetLattice(m_fGenMat,s_fDesRate)

    # Generate lattice points inside the unit ball for vector quantization
    #
    # Syntax
    # -------------------------------------------------------
    # [m_fLattice, m_fGenMat_t] = m_fGetLattice(m_fGenMat,s_fDesRate)
    #
    # INPUT:
    # -------------------------------------------------------
    # m_fGenMat - basic lattice generator matix (matrix)
    # s_fDesRate - desired code rate (positive scalar) 
    #
    # OUTPUT:
    # -------------------------------------------------------
    # m_fLattice  - lattice points (matrix)
    # m_fGenMat_t  - updated generator matrix (matrix)
    
    s_nDim = size(m_fGenMat,1);
    # Number of lattice points in unit ball
    #s_fPoints = (pi^(s_nDim/2))/(gamma(1+(s_nDim/2))*det(m_fGenMat));
    # Number of desired lattice points
    #s_fDesPoints = floor(2^(s_fDesRate*s_nDim + 1));
    # Generate basis lattice - supporting 2-D and 3-D lattices
    s_nMaxDim =  floor(2^(s_fDesRate));
    m_fBaseLattice = zeros(s_nDim,(2*s_nMaxDim + 1)^s_nDim);
    idx = 1;
    for kk=-s_nMaxDim:s_nMaxDim
        for ll=-s_nMaxDim:s_nMaxDim
            m_fBaseLattice[:,idx] = [kk;ll];
            idx = idx+1;
        end
    end
    # Scale generator matrix
    m_fGenMat_t = m_fGenMat/(sqrt(det(m_fGenMat))*s_nMaxDim);
    # Save all resulting lattice points inside the unit cube
    m_fLattice = m_fGenMat_t*m_fBaseLattice;
    m_fLattice = m_fLattice - 2*(m_fLattice .> 1) + 2* (m_fLattice .< -1);
    return m_fLattice, m_fGenMat_t
end

function m_fGenDither(m_fGenMat, s_nSamples)

    # % Generate dither uniformly distrubted over basis cell
    # %
    # % Syntax
    # % -------------------------------------------------------
    # % m_fDither = m_fGenDither(m_fGenMat,s_nSamples)
    # %
    # % INPUT:
    # % -------------------------------------------------------
    # % m_fGenMat - basic lattice generator matix (matrix)
    # % s_nSamples - number of samples to generate (positive scalar) 
    # %
    # % OUTPUT:
    # % -------------------------------------------------------
    # % m_fDither  - dither vectors points (matrix) 
    s_nDim = size(m_fGenMat,1); 
    s_nMaxDim =  1;
    m_fBaseLattice = zeros(s_nDim,(2*s_nMaxDim + 1)^s_nDim);
    idx = 1;
    s_nZeroIdx = -1;
    for kk=-s_nMaxDim:s_nMaxDim
        for ll=-s_nMaxDim:s_nMaxDim
            m_fBaseLattice[:,idx] = [kk;ll];
            if norm(m_fBaseLattice[:,idx])==0
                s_nZeroIdx = idx
            end
            idx = idx+1;
        end
    end
    m_fLattice = m_fGenMat*m_fBaseLattice;
    
    # Randomize points and select those that are closest to center
    MM = rand(s_nDim, 3*s_nSamples)
    m_fPoints = diagm(vec(maximum(abs.(m_fLattice'), dims=1)))*(MM .- 0.5);
    v_fLatIdx =  nearestneighbour(m_fPoints,m_fLattice);
    v_fIdx = filter(x-> v_fLatIdx[x] == s_nZeroIdx, 1:length(v_fLatIdx))  # find(v_fLatIdx==s_nZeroIdx);
    
    if length(v_fIdx) >= s_nSamples
        m_fDither = m_fPoints[:,v_fIdx[1:s_nSamples]];
    else
        m_fDither = [m_fPoints[:,v_fIdx], m_fPoints[:,v_fIdx[1:(s_nSamples-length(v_fIdx))]]];
    end
    return m_fDither
end



function encode_uveqfed(m_fH, s_fDesRate, seed)
    if seed != nothing
        Random.seed!(seed)
    end
    
    # Global variables - used to prevent re-generating lattices repeatedly
    gm_fGenMat2D = [];
    gm_fLattice2D = [];

    # Dithered 2-D lattice quantization
    s_nDim = 2;

    # Reshape into vector
    v_fH = m_fH
    # Zero pad if needed
    s_nExtraZero = mod(length(v_fH),s_nDim)
    if s_nExtraZero != 0
        s_nExtraZero = s_nDim - s_nExtraZero
    end
    v_fH = [v_fH; zeros(s_nExtraZero,1)];
    # Encoder input divided into blocks
    m_fEncInput = reshape(v_fH, s_nDim, :);
    # Scaling of the standard deviation - experimental value
    # Standard deviation scaling
    s_fRatio = 2 + s_fDesRate/5;
    s_fScale= s_fRatio*sqrt(sum(std(m_fEncInput', dims=1).^2) + 1e-10);
    # Dithered lattice quantization
    m_fEncInput = m_fEncInput/s_fScale;
    # Generate lattices if not previously generated
    if isempty(gm_fLattice2D)
        # Hexagonal lattice basic generator matrix
        m_fGenMat =  [2 1; 0 sqrt(3)]#[2, 0; 1, sqrt(3)]';
        gm_fLattice2D, gm_fGenMat2D = m_fGetLattice(m_fGenMat,s_fDesRate);
    end
    m_fGenMat_t = gm_fGenMat2D;
    m_fLattice = gm_fLattice2D;
    
    # Generate dither
    m_fDither = m_fGenDither(m_fGenMat_t,size(m_fEncInput,2)); 

    # Quantize
    v_fLatIdx =  nearestneighbour((m_fEncInput + m_fDither),m_fLattice);
    databytes = Vector{UInt8}()

    append!(databytes, reinterpret(UInt8, [Float32(s_fScale)]))
    append!(databytes, reinterpret(UInt8, [UInt32(s_nExtraZero)]))
    for idx in v_fLatIdx
        append!(databytes, reinterpret(UInt8, [Int16(idx-1)]))
    end

    counts = zeros(maximum(v_fLatIdx))
    for num in v_fLatIdx
        counts[num] += 1
    end
    cdf = zeros(length(counts)+1)
    for num in 2:length(cdf)
        cdf[num] = cdf[num-1] + counts[num-1]
    end
    cdf = cdf./cdf[end]
    return databytes, Int16.(v_fLatIdx.-1), cdf
end

function decode_uveqfed(databytes, seed, s_fDesRate)
    Random.seed!(seed)

    m_fLattice, m_fGenMat_t = m_fGetLattice([2 1; 0 sqrt(3)],s_fDesRate);

    databytes = Vector{UInt8}(join(databytes))
    s_fScale = reinterpret(Float32, databytes[1:4])[1]
    s_nExtraZero = reinterpret(UInt32, databytes[5:8])[1]
    v_fLatIdx = reinterpret(Int16, databytes[9:end]).+1
    
    m_fQ =  m_fLattice[:,v_fLatIdx];
    m_fDither = m_fGenDither(m_fGenMat_t,size(m_fQ,2)); 

    # Decode by projecting back from lattice and substracting dither
    m_fDecOutput = s_fScale*(m_fQ - m_fDither);

    # Remove zero padding
    v_fDecOutput = m_fDecOutput # (:);
    v_fDecOutput = v_fDecOutput[1:end-s_nExtraZero];
    # Re-scale
    m_fHhat = reshape(v_fDecOutput, :);

    return m_fHhat
end
