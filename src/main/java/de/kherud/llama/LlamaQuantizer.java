package de.kherud.llama;

public class LlamaQuantizer {

	static {
		LlamaLoader.initialize();
	}

	public static class QuantizationParams {
		public int nthread = 0; // Default: use hardware_concurrency
		public int ftype = QuantizationType.MOSTLY_Q4_0.getValue(); // Default to Q4_0
		public boolean allowRequantize = false;
		public boolean quantizeOutputTensor = true;
		public boolean onlyCopy = false;
		public boolean pure = false;
		public boolean keepSplit = false;

		public QuantizationParams() {
		}

		public QuantizationParams setThreadCount(int nthread) {
			this.nthread = nthread;
			return this;
		}

		public QuantizationParams setQuantizationType(QuantizationType ftype) {
			this.ftype = ftype.getValue();
			return this;
		}

		public QuantizationParams setAllowRequantize(boolean allowRequantize) {
			this.allowRequantize = allowRequantize;
			return this;
		}

		public QuantizationParams setQuantizeOutputTensor(boolean quantizeOutputTensor) {
			this.quantizeOutputTensor = quantizeOutputTensor;
			return this;
		}

		public QuantizationParams setOnlyCopy(boolean onlyCopy) {
			this.onlyCopy = onlyCopy;
			return this;
		}

		public QuantizationParams setPure(boolean pure) {
			this.pure = pure;
			return this;
		}

		public QuantizationParams setKeepSplit(boolean keepSplit) {
			this.keepSplit = keepSplit;
			return this;
		}

		public int getThreadCount() {
			return nthread;
		}

		public QuantizationType getQuantizationType() {
			return QuantizationType.fromValue(ftype);
		}

		public boolean isAllowRequantize() {
			return allowRequantize;
		}

		public boolean isQuantizeOutputTensor() {
			return quantizeOutputTensor;
		}

		public boolean isOnlyCopy() {
			return onlyCopy;
		}

		public boolean isPure() {
			return pure;
		}

		public boolean isKeepSplit() {
			return keepSplit;
		}
	}

	public enum QuantizationType {
		ALL_F32(0, "All F32"),
		MOSTLY_F16(1, "Mostly F16"),
		MOSTLY_Q4_0(2, "Q4_0"),
		MOSTLY_Q4_1(3, "Q4_1"),
		MOSTLY_Q8_0(7, "Q8_0"),
		MOSTLY_Q5_0(8, "Q5_0"),
		MOSTLY_Q5_1(9, "Q5_1"),
		MOSTLY_Q2_K(10, "Q2_K"),
		MOSTLY_Q3_K_S(11, "Q3_K_S"),
		MOSTLY_Q3_K_M(12, "Q3_K_M"),
		MOSTLY_Q3_K_L(13, "Q3_K_L"),
		MOSTLY_Q4_K_S(14, "Q4_K_S"),
		MOSTLY_Q4_K_M(15, "Q4_K_M"),
		MOSTLY_Q5_K_S(16, "Q5_K_S"),
		MOSTLY_Q5_K_M(17, "Q5_K_M"),
		MOSTLY_Q6_K(18, "Q6_K"),
		MOSTLY_IQ2_XXS(19, "IQ2_XXS"),
		MOSTLY_IQ2_XS(20, "IQ2_XS"),
		MOSTLY_Q2_K_S(21, "Q2_K_S"),
		MOSTLY_IQ3_XS(22, "IQ3_XS"),
		MOSTLY_IQ3_XXS(23, "IQ3_XXS"),
		MOSTLY_IQ1_S(24, "IQ1_S"),
		MOSTLY_IQ4_NL(25, "IQ4_NL"),
		MOSTLY_IQ3_S(26, "IQ3_S"),
		MOSTLY_IQ3_M(27, "IQ3_M"),
		MOSTLY_IQ2_S(28, "IQ2_S"),
		MOSTLY_IQ2_M(29, "IQ2_M"),
		MOSTLY_IQ4_XS(30, "IQ4_XS"),
		MOSTLY_IQ1_M(31, "IQ1_M"),
		MOSTLY_BF16(32, "BF16"),
		MOSTLY_TQ1_0(36, "TQ1_0"),
		MOSTLY_TQ2_0(37, "TQ2_0"),
		MOSTLY_MXFP4_MOE(38, "MXFP4_MOE");

		private final int value;
		private final String description;

		QuantizationType(int value, String description) {
			this.value = value;
			this.description = description;
		}

		public int getValue() {
			return value;
		}

		public String getDescription() {
			return description;
		}

		public static QuantizationType fromValue(int value) {
			for (QuantizationType type : values()) {
				if (type.value == value) {
					return type;
				}
			}
			throw new IllegalArgumentException("Unknown quantization type: " + value);
		}

		@Override
		public String toString() {
			return description + " (" + value + ")";
		}
	}

	public static QuantizationParams getDefaultParams() throws LlamaException {
		return getDefaultQuantizationParamsNative();
	}

	public static void quantizeModel(String inputPath, String outputPath) throws LlamaException {
		quantizeModel(inputPath, outputPath, getDefaultParams());
	}

	public static void quantizeModel(String inputPath, String outputPath, QuantizationType type) throws LlamaException {
		QuantizationParams params = getDefaultParams();
		params.setQuantizationType(type);
		quantizeModel(inputPath, outputPath, params);
	}

	public static void quantizeModel(String inputPath, String outputPath, QuantizationParams params) throws LlamaException {
		if (inputPath == null || inputPath.trim().isEmpty()) {
			throw new IllegalArgumentException("Input path cannot be null or empty");
		}
		if (outputPath == null || outputPath.trim().isEmpty()) {
			throw new IllegalArgumentException("Output path cannot be null or empty");
		}
		if (params == null) {
			params = getDefaultParams();
		}

		int result = quantizeModelNative(inputPath, outputPath, params);
		if (result != 0) {
			throw new LlamaException("Model quantization failed with error code: " + result);
		}
	}

	private static native QuantizationParams getDefaultQuantizationParamsNative();

	private static native int quantizeModelNative(String inputPath, String outputPath, QuantizationParams params);
}