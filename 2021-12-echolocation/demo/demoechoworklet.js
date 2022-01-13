const renderQuantum = 128;

/**
 * Multiple delay/gain lines mapping mono input to stereo output.
 */
class EchoProcessor extends AudioWorkletProcessor {
    constructor(options) {
        super(options);
        this.echoes = options.processorOptions.echoes.map(echo => {
            return [
                Math.round(sampleRate * echo.delayLeft),
                echo.gainLeft,
                Math.round(sampleRate * echo.delayRight),
                echo.gainRight,
            ];
        });
        this.maxDelay = this.echoes.reduce(
            (max, [decayLeft, _1, decayRight, _2]) => Math.max(max, decayLeft, decayRight),
            0);
        this.bufferLength = this.maxDelay + renderQuantum + 1;
        this.leftBuffer = new Float32Array(this.bufferLength);
        this.rightBuffer = new Float32Array(this.bufferLength);
        this.readIdx = 0;
        this.lastWriteIdx = 0;
    }

    process(inputs, outputs) {
        if (inputs[0].length) {
            const input = inputs[0][0];
            for (const [delayLeft, gainLeft, delayRight, gainRight] of this.echoes) {
                for (let i = 0; i < renderQuantum; ++i) {
                    this.leftBuffer[(this.readIdx + i + delayLeft) % this.bufferLength] += gainLeft * input[i];
                    this.rightBuffer[(this.readIdx + i + delayRight) % this.bufferLength] += gainRight * input[i];
                }
            }
            this.lastWriteIdx = (this.readIdx + renderQuantum + this.maxDelay) % this.bufferLength;
        }
        const outputLeft = outputs[0][0];
        const outputRight = outputs[0][1];
        for (let i = 0; i < renderQuantum && this.readIdx !== this.lastWriteIdx; ++i) {
            outputLeft[i] = this.leftBuffer[this.readIdx];
            outputRight[i] = this.rightBuffer[this.readIdx];
            this.leftBuffer[this.readIdx] = 0;
            this.rightBuffer[this.readIdx] = 0;
            this.readIdx = (this.readIdx + 1) % this.bufferLength;
        }
        return this.readIdx !== this.lastWriteIdx;
    }
}

registerProcessor("echo-processor", EchoProcessor);
