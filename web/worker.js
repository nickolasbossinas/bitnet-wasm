/*
 * BitNet Inference Web Worker
 *
 * Runs the WASM inference module off the main thread.
 * Communicates via postMessage:
 *
 * Inbound:
 *   {type: 'load', buffer: ArrayBuffer, n_layers: number}
 *   {type: 'generate', prompt, max_tokens, temperature, top_p, seed}
 *
 * Outbound:
 *   {type: 'ready'}
 *   {type: 'progress', message: string}
 *   {type: 'loaded', success: boolean, error?: string}
 *   {type: 'token', piece: string, id: number}   (from C via EM_ASM)
 *   {type: 'stats', text: string}
 *   {type: 'done', n_tokens: number}
 *   {type: 'error', message: string}
 */

importScripts('bitnet-inference.js');

let Module = null;

async function initModule() {
    try {
        Module = await BitNetInference({
            print: function(text) {
                self.postMessage({ type: 'progress', message: text });
            },
            printErr: function(text) {
                self.postMessage({ type: 'stats', text: text });
            }
        });
        self.postMessage({ type: 'ready' });
    } catch (e) {
        self.postMessage({ type: 'error', message: 'Failed to load WASM module: ' + e.message });
    }
}

async function loadModel(buffer, nLayers) {
    try {
        const size = buffer.byteLength;
        self.postMessage({
            type: 'progress',
            message: 'Allocating ' + (size / 1024 / 1024).toFixed(0) + ' MB in WASM heap...'
        });

        /* Allocate WASM heap memory and copy file data */
        const ptr = Module._malloc(size);
        if (!ptr) {
            self.postMessage({
                type: 'loaded',
                success: false,
                error: 'Failed to allocate ' + (size / 1024 / 1024).toFixed(0) + ' MB in WASM heap'
            });
            return;
        }

        Module.HEAPU8.set(new Uint8Array(buffer), ptr);
        buffer = null; /* hint GC to release the ArrayBuffer */

        self.postMessage({ type: 'progress', message: 'Parsing GGUF and loading weights...' });

        var result = Module._bitnet_init(ptr, size, nLayers);

        /* Free the GGUF file buffer — weights are copied out */
        Module._free(ptr);

        if (result === 0) {
            self.postMessage({ type: 'loaded', success: true });
        } else {
            var errors = {
                '-1': 'GGUF parse failed',
                '-2': 'Tokenizer init failed',
                '-3': 'Model alloc failed (out of memory?)',
                '-4': 'Weight loading failed',
                '-5': 'Already initialized'
            };
            self.postMessage({
                type: 'loaded',
                success: false,
                error: errors[String(result)] || 'Unknown error (code ' + result + ')'
            });
        }
    } catch (e) {
        self.postMessage({
            type: 'loaded',
            success: false,
            error: e.message
        });
    }
}

function runGenerate(prompt, maxTokens, temperature, topP, seed) {
    try {
        var promptPtr = Module.stringToNewUTF8(prompt);
        var n = Module._bitnet_generate(promptPtr, maxTokens, temperature, topP, seed);
        Module._free(promptPtr);
        self.postMessage({ type: 'done', n_tokens: n });
    } catch (e) {
        self.postMessage({ type: 'error', message: e.message });
    }
}

self.onmessage = function(e) {
    var msg = e.data;
    switch (msg.type) {
        case 'load':
            loadModel(msg.buffer, msg.n_layers || -1);
            break;
        case 'generate':
            runGenerate(
                msg.prompt,
                msg.max_tokens || 64,
                msg.temperature || 0.0,
                msg.top_p || 0.9,
                msg.seed || 42
            );
            break;
    }
};

/* Initialize WASM module on worker startup */
initModule();
