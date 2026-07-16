/* ═══════════════════════════════════════════════════════════════════
   Word Forge — Search (Autocomplete)
   ═══════════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    class Search {
        /**
         * @param {HTMLInputElement} inputEl
         * @param {HTMLElement} resultsEl
         * @param {object} opts
         * @param {function} opts.onSelect — called with the selected result object
         */
        constructor(inputEl, resultsEl, opts) {
            this.inputEl = inputEl;
            this.resultsEl = resultsEl;
            this.onSelect = (opts && opts.onSelect) || function () {};
            this.debounceMs = 300;
            this._timer = null;
            this._activeIndex = -1;
            this._items = [];
            this._abortController = null;

            this._bindEvents();
        }

        /* ─── Event Binding ─── */
        _bindEvents() {
            this.inputEl.addEventListener('input', () => this._onInput());
            this.inputEl.addEventListener('keydown', (e) => this._onKeyDown(e));

            // Close on outside click
            document.addEventListener('mousedown', (e) => {
                if (!this.resultsEl.contains(e.target) && e.target !== this.inputEl) {
                    this.close();
                }
            });
        }

        /* ─── Input Handler (debounced) ─── */
        _onInput() {
            clearTimeout(this._timer);
            const query = this.inputEl.value.trim();

            if (query.length < 1) {
                this.close();
                return;
            }

            this._timer = setTimeout(() => this._fetch(query), this.debounceMs);
        }

        /* ─── Keyboard Navigation ─── */
        _onKeyDown(e) {
            if (!this.resultsEl.classList.contains('is-open')) {
                if (e.key === 'Enter') {
                    // Enter without dropdown → fire onSelect with raw input
                    e.preventDefault();
                    const term = this.inputEl.value.trim();
                    if (term) {
                        this.close();
                        this.onSelect({ term: term, language: 'en' });
                    }
                }
                return;
            }

            switch (e.key) {
                case 'ArrowDown':
                    e.preventDefault();
                    this._moveActive(1);
                    break;
                case 'ArrowUp':
                    e.preventDefault();
                    this._moveActive(-1);
                    break;
                case 'Enter':
                    e.preventDefault();
                    if (this._activeIndex >= 0 && this._items[this._activeIndex]) {
                        this._selectItem(this._items[this._activeIndex]);
                    } else {
                        const term = this.inputEl.value.trim();
                        if (term) {
                            this.close();
                            this.onSelect({ term: term, language: 'en' });
                        }
                    }
                    break;
                case 'Escape':
                    e.preventDefault();
                    this.close();
                    break;
            }
        }

        _moveActive(delta) {
            const children = this.resultsEl.querySelectorAll('.wf-search-item');
            if (!children.length) return;

            // Remove current
            if (children[this._activeIndex]) {
                children[this._activeIndex].classList.remove('is-active');
            }

            this._activeIndex += delta;
            if (this._activeIndex < 0) this._activeIndex = children.length - 1;
            if (this._activeIndex >= children.length) this._activeIndex = 0;

            children[this._activeIndex].classList.add('is-active');
            children[this._activeIndex].scrollIntoView({ block: 'nearest' });
        }

        /* ─── Fetch Search Results ─── */
        async _fetch(query) {
            // Abort previous in-flight request
            if (this._abortController) {
                this._abortController.abort();
            }
            this._abortController = new AbortController();

            try {
                const resp = await fetch(
                    '/api/words/search?q=' + encodeURIComponent(query) + '&limit=10',
                    { signal: this._abortController.signal }
                );

                if (!resp.ok) throw new Error('Search failed');

                const data = await resp.json();
                this._items = data.results || [];
                this._render();
            } catch (err) {
                if (err.name === 'AbortError') return; // Expected, ignore
                console.warn('[Search] Fetch error:', err);
                this.close();
            }
        }

        /* ─── Render Dropdown ─── */
        _render() {
            if (!this._items.length) {
                this.resultsEl.innerHTML = '<div class="wf-search-item"><span class="wf-search-item__term" style="color:var(--wf-text-dim);font-style:italic">No results</span></div>';
                this.resultsEl.classList.add('is-open');
                this._activeIndex = -1;
                return;
            }

            this.resultsEl.innerHTML = this._items.map((item, i) => {
                const pronIcon = item.has_pronunciation ? '<span class="has" title="Has pronunciation">🔊</span>' : '';
                const emoIcon = item.has_emotion ? '<span class="has" title="Has emotion data">❤</span>' : '';
                const stubMark = item.is_stub ? '<span title="Stub entry" style="color:var(--wf-warning)">◇</span>' : '';

                return '<div class="wf-search-item" data-index="' + i + '" role="option">' +
                    '<span class="wf-search-item__term">' + this._escHtml(item.term) + '</span>' +
                    stubMark +
                    '<span class="wf-search-item__lang">' + this._escHtml(item.language || 'en') + '</span>' +
                    '<span class="wf-search-item__icons">' + pronIcon + emoIcon + '</span>' +
                '</div>';
            }).join('');

            this.resultsEl.classList.add('is-open');
            this._activeIndex = -1;

            // Bind click handlers
            this.resultsEl.querySelectorAll('.wf-search-item').forEach((el) => {
                el.addEventListener('mousedown', (e) => {
                    e.preventDefault(); // Prevent input blur
                    const idx = parseInt(el.dataset.index, 10);
                    if (this._items[idx]) {
                        this._selectItem(this._items[idx]);
                    }
                });
            });
        }

        /* ─── Select Item ─── */
        _selectItem(item) {
            this.inputEl.value = item.term;
            this.close();
            this.onSelect(item);
        }

        /* ─── Close Dropdown ─── */
        close() {
            this.resultsEl.classList.remove('is-open');
            this.resultsEl.innerHTML = '';
            this._activeIndex = -1;
        }

        /* ─── Set Input Value Programmatically ─── */
        setValue(term) {
            this.inputEl.value = term;
            this.close();
        }

        /* ─── Helpers ─── */
        _escHtml(str) {
            var d = document.createElement('div');
            d.textContent = str;
            return d.innerHTML;
        }
    }

    // Export
    window.Search = Search;
})();
