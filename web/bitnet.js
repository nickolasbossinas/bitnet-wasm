/**
 * BitNet.js — Integration API for BitNet WASM inference.
 *
 * Usage:
 *   import { BitNet } from './bitnet.js';
 *
 *   const bitnet = new BitNet();
 *   await bitnet.load('model.gguf', { threads: 5 });
 *   const result = await bitnet.generate('Hello', {
 *       maxTokens: 64,
 *       onToken: (piece) => process.stdout.write(piece)
 *   });
 *   console.log(result.text);
 *   bitnet.destroy();
 */

export class BitNet {
    #worker = null;
    #workerUrl;
    #loaded = false;
    #config = null;
    #generating = false;

    /* Message routing */
    #onReady = null;
    #onLoaded = null;
    #onThreadsSet = null;
    #onConfig = null;
    #onToken = null;
    #onDone = null;
    #onProgress = null;
    #statsLines = [];
    #textPieces = [];

    /**
     * @param {Object} [options]
     * @param {string} [options.workerUrl='./worker.js'] Path to worker.js
     */
    constructor(options = {}) {
        this.#workerUrl = options.workerUrl || './worker.js';
    }

    /** Whether a model is loaded and ready for generation. */
    get loaded() { return this.#loaded; }

    /** Model configuration (null if not loaded). */
    get config() { return this.#config; }

    /** Whether generation is currently in progress. */
    get generating() { return this.#generating; }

    /**
     * Load a model from a URL or ArrayBuffer.
     *
     * @param {string|ArrayBuffer} source  URL to fetch or raw GGUF data
     * @param {Object} [options]
     * @param {number} [options.threads=0]  Worker threads (0 = single-threaded)
     * @param {number} [options.layers=-1]  Layers to load (-1 = all)
     * @param {function} [options.onProgress] Progress callback: ({phase, message, percent?})
     * @returns {Promise<Object>} Model config object
     */
    async load(source, options = {}) {
        if (this.#loaded) {
            throw new Error('Model already loaded. Call destroy() first.');
        }

        const threads = options.threads || 0;
        const layers = options.layers || -1;
        const onProgress = options.onProgress || null;

        try {
        /* 1. Create worker and wait for WASM ready */
        this.#worker = new Worker(this.#workerUrl);
        this.#worker.onmessage = (e) => this.#handleMessage(e.data);

        await new Promise((resolve, reject) => {
            this.#onReady = resolve;
            setTimeout(() => reject(new Error('Worker init timeout')), 30000);
        });

        /* 2. Fetch GGUF if source is URL */
        let buffer;
        if (typeof source === 'string') {
            if (onProgress) onProgress({ phase: 'download', message: 'Fetching model...' });

            const response = await fetch(source);
            if (!response.ok) {
                throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
            }

            const contentLength = parseInt(response.headers.get('content-length') || '0', 10);
            if (contentLength && onProgress) {
                /* Stream with progress */
                const reader = response.body.getReader();
                const chunks = [];
                let received = 0;
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    chunks.push(value);
                    received += value.length;
                    onProgress({
                        phase: 'download',
                        message: `Downloading: ${(received / 1024 / 1024).toFixed(0)} / ${(contentLength / 1024 / 1024).toFixed(0)} MB`,
                        percent: Math.round(received / contentLength * 100)
                    });
                }
                buffer = new Uint8Array(received);
                let offset = 0;
                for (const chunk of chunks) {
                    buffer.set(chunk, offset);
                    offset += chunk.length;
                }
                buffer = buffer.buffer;
            } else {
                buffer = await response.arrayBuffer();
            }
        } else if (source instanceof ArrayBuffer) {
            buffer = source;
        } else {
            throw new Error('source must be a URL string or ArrayBuffer');
        }

        /* 3. Transfer buffer to worker and load model */
        if (onProgress) {
            this.#onProgress = onProgress;
        }

        const loadResult = await new Promise((resolve, reject) => {
            this.#onLoaded = { resolve, reject };
            this.#worker.postMessage(
                { type: 'load', buffer: buffer, n_layers: layers, n_threads: threads },
                [buffer]
            );
        });
        this.#onProgress = null;

        if (!loadResult.success) {
            throw new Error(loadResult.error || 'Model load failed');
        }

        /* 4. Query model config */
        this.#config = await new Promise((resolve) => {
            this.#onConfig = resolve;
            this.#worker.postMessage({ type: 'get_config' });
        });

        this.#loaded = true;
        return this.#config;

        } catch (e) {
            /* Clean up on failure so load() can be retried */
            if (this.#worker) {
                this.#worker.terminate();
                this.#worker = null;
            }
            this.#onProgress = null;
            throw e;
        }
    }

    /**
     * Generate text from a prompt.
     *
     * @param {string} prompt  Input text
     * @param {Object} [options]
     * @param {number} [options.maxTokens=64]
     * @param {number} [options.temperature=0.0]  0 = greedy
     * @param {number} [options.topP=0.9]
     * @param {number} [options.seed=42]
     * @param {number} [options.repetitionPenalty=1.1]
     * @param {function} [options.onToken]  Called with each text piece: (piece: string) => void
     * @param {AbortSignal} [options.signal]  AbortSignal for cancellation
     * @returns {Promise<{text: string, tokenCount: number, stats: Object}>}
     */
    generate(prompt, options = {}) {
        if (!this.#loaded) {
            return Promise.reject(new Error('Model not loaded. Call load() first.'));
        }
        if (this.#generating) {
            return Promise.reject(new Error('Generation already in progress.'));
        }

        this.#generating = true;
        this.#textPieces = [];
        this.#statsLines = [];
        this.#onToken = options.onToken || null;

        return new Promise((resolve, reject) => {
            /* Abort signal handling */
            if (options.signal) {
                if (options.signal.aborted) {
                    this.#generating = false;
                    reject(new DOMException('Aborted', 'AbortError'));
                    return;
                }
                options.signal.addEventListener('abort', () => {
                    this.abort();
                    this.#generating = false;
                    reject(new DOMException('Aborted', 'AbortError'));
                }, { once: true });
            }

            this.#onDone = (nTokens) => {
                this.#generating = false;
                resolve({
                    text: this.#textPieces.join(''),
                    tokenCount: nTokens,
                    stats: this.#parseStats()
                });
            };

            this.#worker.postMessage({
                type: 'generate',
                prompt: prompt,
                max_tokens: options.maxTokens || 64,
                temperature: options.temperature || 0.0,
                top_p: options.topP || 0.9,
                seed: options.seed || 42,
                repetition_penalty: options.repetitionPenalty || 1.1
            });
        });
    }

    /**
     * Change the number of worker threads.
     * @param {number} n  Number of worker threads (0 = single-threaded)
     * @returns {Promise<void>}
     */
    async setThreads(n) {
        if (!this.#loaded) throw new Error('Model not loaded.');

        await new Promise((resolve, reject) => {
            this.#onThreadsSet = { resolve, reject };
            this.#worker.postMessage({ type: 'set_threads', n_threads: n });
        });

        /* Refresh config to reflect new thread count */
        this.#config = await new Promise((resolve) => {
            this.#onConfig = resolve;
            this.#worker.postMessage({ type: 'get_config' });
        });
    }

    /** Abort in-flight generation. */
    abort() {
        if (this.#worker) {
            this.#worker.postMessage({ type: 'abort' });
        }
    }

    /** Release all resources and terminate the worker. */
    destroy() {
        if (this.#worker) {
            if (this.#loaded) {
                this.#worker.postMessage({ type: 'free' });
            }
            this.#worker.terminate();
            this.#worker = null;
        }
        this.#loaded = false;
        this.#config = null;
        this.#generating = false;
    }

    /* --- Internal --- */

    #handleMessage(msg) {
        switch (msg.type) {
            case 'ready':
                if (this.#onReady) { this.#onReady(); this.#onReady = null; }
                break;

            case 'progress':
                if (this.#onProgress) {
                    this.#onProgress({ phase: 'load', message: msg.message });
                }
                break;

            case 'loaded':
                if (this.#onLoaded) {
                    this.#onLoaded.resolve(msg);
                    this.#onLoaded = null;
                }
                break;

            case 'config':
                if (this.#onConfig) { this.#onConfig(msg.config); this.#onConfig = null; }
                break;

            case 'threads_set':
                if (this.#onThreadsSet) {
                    if (msg.success) {
                        this.#onThreadsSet.resolve();
                    } else {
                        this.#onThreadsSet.reject(new Error(msg.error || 'set_threads failed'));
                    }
                    this.#onThreadsSet = null;
                }
                break;

            case 'token':
                this.#textPieces.push(msg.piece);
                if (this.#onToken) this.#onToken(msg.piece);
                break;

            case 'stats':
                this.#statsLines.push(msg.text);
                break;

            case 'done':
                if (this.#onDone) { this.#onDone(msg.n_tokens); this.#onDone = null; }
                break;

            case 'error':
                /* Route errors to the appropriate pending promise */
                if (this.#onLoaded) {
                    this.#onLoaded.resolve({ success: false, error: msg.message });
                    this.#onLoaded = null;
                }
                break;
        }
    }

    #parseStats() {
        const stats = {};
        for (const line of this.#statsLines) {
            let m;
            if ((m = line.match(/\[generate\] prefill:\s*(\d+) tokens,\s*([\d.]+) ms\s*\(([\d.]+) tok\/s\)/))) {
                stats.prefillTokens = parseInt(m[1]);
                stats.prefillMs = parseFloat(m[2]);
                stats.prefillTokPerSec = parseFloat(m[3]);
            }
            if ((m = line.match(/\[generate\] decode:\s*(\d+) tokens,\s*([\d.]+) ms\s*\(([\d.]+) tok\/s\)/))) {
                stats.decodeTokens = parseInt(m[1]);
                stats.decodeMs = parseFloat(m[2]);
                stats.decodeTokPerSec = parseFloat(m[3]);
            }
            if ((m = line.match(/\[generate\] total:\s*([\d.]+) ms/))) {
                stats.totalMs = parseFloat(m[1]);
            }
        }
        return stats;
    }
}
