/* ═══════════════════════════════════════════════════════════════════
   Word Forge — Inspector (Word Detail Panel)
   ═══════════════════════════════════════════════════════════════════ */

(function () {
    'use strict';

    class Inspector {
        constructor() {
            this.panel = document.getElementById('wf-inspector');
            this.closeBtn = document.getElementById('wf-inspector-close');
            this.loadingEl = document.getElementById('wf-inspector-loading');
            this.contentEl = document.getElementById('wf-inspector-content');
            this._currentTerm = null;
            this._activeRelTab = null;
            this._relData = null;
            this._navigateCb = null; // callback for term navigation

            this._bindEvents();
        }

        _bindEvents() {
            var self = this;
            if (this.closeBtn) {
                this.closeBtn.addEventListener('click', function () { self.close(); });
            }
        }

        /**
         * Register callback invoked when user clicks a related term.
         * @param {function} cb — receives (term)
         */
        onNavigate(cb) {
            this._navigateCb = cb;
        }

        /**
         * Open the inspector for a given term.
         * @param {string} term
         */
        async open(term) {
            if (!term) return;
            this._currentTerm = term;

            // Show panel + loading
            this.panel.classList.add('is-open');
            this.panel.setAttribute('aria-hidden', 'false');
            this._showLoading(true);
            this.contentEl.classList.add('is-hidden');

            try {
                var resp = await fetch('/api/words/' + encodeURIComponent(term));
                if (!resp.ok) throw new Error('Word not found (HTTP ' + resp.status + ')');
                var data = await resp.json();
                this._render(data);
            } catch (err) {
                this._renderError(err.message);
            } finally {
                this._showLoading(false);
                this.contentEl.classList.remove('is-hidden');
            }
        }

        /**
         * Close the inspector panel.
         */
        close() {
            this.panel.classList.remove('is-open');
            this.panel.setAttribute('aria-hidden', 'true');
            this._currentTerm = null;
        }

        /**
         * Check if inspector is currently open.
         * @returns {boolean}
         */
        isOpen() {
            return this.panel.classList.contains('is-open');
        }

        /* ─── Rendering ─── */

        _render(data) {
            this.renderHeader(data);
            this.renderDefinition(data);
            this.renderPronunciations(data);
            this.renderEmotion(data);
            this.renderRelationships(data);
            this.renderGraphemes(data);
            this.renderExamples(data);
        }

        _renderError(message) {
            this.contentEl.innerHTML =
                '<div class="wf-insp-section">' +
                '<p style="color:var(--wf-danger);font-size:14px;">⚠ ' + this._esc(message) + '</p>' +
                '</div>';
        }

        /* ─── Header ─── */
        renderHeader(data) {
            var el = document.getElementById('wf-insp-header');
            var posBadge = data.part_of_speech
                ? '<span class="wf-badge wf-badge--pos">' + this._esc(data.part_of_speech) + '</span>'
                : '';
            var langBadge = '<span class="wf-badge wf-badge--lang">' + this._esc(data.language || 'en') + '</span>';
            var stubBadge = data.is_stub ? '<span class="wf-badge wf-badge--stub">Stub</span>' : '';

            el.innerHTML =
                '<div class="wf-insp-header__term">' + this._esc(data.term || '') + '</div>' +
                '<div class="wf-insp-header__meta">' + posBadge + langBadge + stubBadge + '</div>';
            el.classList.remove('is-hidden');
        }

        /* ─── Definition ─── */
        renderDefinition(data) {
            var el = document.getElementById('wf-insp-definition');
            if (data.definition) {
                el.innerHTML = '<p class="wf-insp-definition__text">' + this._esc(data.definition) + '</p>';
            } else {
                el.innerHTML = '<p class="wf-insp-definition__empty">No definition available</p>';
            }
            el.classList.remove('is-hidden');
        }

        /* ─── Pronunciations ─── */
        renderPronunciations(data) {
            var section = document.getElementById('wf-insp-pronunciations');
            var list = document.getElementById('wf-insp-pron-list');

            if (!data.pronunciations || data.pronunciations.length === 0) {
                section.classList.add('is-hidden');
                return;
            }

            section.classList.remove('is-hidden');
            var html = '';

            data.pronunciations.forEach(function (pron) {
                html += '<div class="wf-pron-entry">';

                // Header: notation badge + transcription + dialect
                html += '<div class="wf-pron-entry__header">';
                html += '<span class="wf-pron-notation">' + this._esc(pron.notation || 'IPA') + '</span>';
                html += '<span class="wf-pron-transcription">/' + this._esc(pron.transcription || '') + '/</span>';
                if (pron.dialect) {
                    html += '<span class="wf-pron-dialect">' + this._esc(pron.dialect) + '</span>';
                }
                html += '</div>';

                // Phoneme chips
                if (pron.phonemes && pron.phonemes.length > 0) {
                    html += '<div class="wf-pron-phonemes">';
                    pron.phonemes.forEach(function (ph) {
                        var stressClass = 'wf-phoneme-chip--unstressed';
                        if (ph.stress === 1) stressClass = 'wf-phoneme-chip--primary';
                        else if (ph.stress === 2) stressClass = 'wf-phoneme-chip--secondary';

                        html += '<span class="wf-phoneme-chip ' + stressClass + '" title="' +
                            this._esc(ph.base_symbol || ph.symbol) +
                            (ph.syllabic ? ' (syllabic)' : '') +
                            '">' + this._esc(ph.symbol) + '</span>';
                    }.bind(this));
                    html += '</div>';
                }

                // Stress pattern dots
                if (pron.stress_pattern && pron.stress_pattern.length > 0) {
                    html += '<div class="wf-pron-stress">';
                    html += '<span class="wf-pron-stress__label">Stress:</span>';
                    pron.stress_pattern.forEach(function (s) {
                        if (s === 1) {
                            html += '<span class="wf-stress-dot wf-stress-dot--primary" title="Primary stress">●</span>';
                        } else if (s === 2) {
                            html += '<span class="wf-stress-dot wf-stress-dot--secondary" title="Secondary stress">●</span>';
                        } else {
                            html += '<span class="wf-stress-dot wf-stress-dot--unstressed" title="Unstressed">○</span>';
                        }
                    });
                    html += '</div>';
                }

                // Meta: syllable count, source
                var metaParts = [];
                if (pron.syllable_count != null) metaParts.push(pron.syllable_count + ' syllable' + (pron.syllable_count !== 1 ? 's' : ''));
                if (pron.source) metaParts.push('Source: ' + pron.source);
                if (metaParts.length) {
                    html += '<div class="wf-pron-meta">' + this._esc(metaParts.join(' · ')) + '</div>';
                }

                html += '</div>'; // .wf-pron-entry
            }.bind(this));

            list.innerHTML = html;
        }

        /* ─── Emotion ─── */
        renderEmotion(data) {
            var section = document.getElementById('wf-insp-emotion');
            var content = document.getElementById('wf-insp-emotion-content');

            if (!data.emotion || (data.emotion.valence == null && data.emotion.arousal == null)) {
                section.classList.add('is-hidden');
                return;
            }

            section.classList.remove('is-hidden');
            var html = '';

            if (data.emotion.valence != null) {
                var vPct = ((data.emotion.valence + 1) / 2 * 100).toFixed(1); // normalize -1..1 → 0..100
                // If valence is 0..1, adjust
                if (data.emotion.valence >= 0 && data.emotion.valence <= 1) {
                    vPct = (data.emotion.valence * 100).toFixed(1);
                }

                html += '<div class="wf-emotion-gauge">';
                html += '<div class="wf-emotion-gauge__header">';
                html += '<span class="wf-emotion-gauge__label">Valence</span>';
                html += '<span class="wf-emotion-gauge__value">' + data.emotion.valence.toFixed(2) + '</span>';
                html += '</div>';
                html += '<div class="wf-emotion-gauge__track wf-emotion-gauge__track--valence">';
                html += '<div class="wf-emotion-gauge__needle" style="left:' + vPct + '%"></div>';
                html += '</div>';
                html += '</div>';
            }

            if (data.emotion.arousal != null) {
                var aPct = ((data.emotion.arousal + 1) / 2 * 100).toFixed(1);
                if (data.emotion.arousal >= 0 && data.emotion.arousal <= 1) {
                    aPct = (data.emotion.arousal * 100).toFixed(1);
                }

                html += '<div class="wf-emotion-gauge">';
                html += '<div class="wf-emotion-gauge__header">';
                html += '<span class="wf-emotion-gauge__label">Arousal</span>';
                html += '<span class="wf-emotion-gauge__value">' + data.emotion.arousal.toFixed(2) + '</span>';
                html += '</div>';
                html += '<div class="wf-emotion-gauge__track wf-emotion-gauge__track--arousal">';
                html += '<div class="wf-emotion-gauge__needle" style="left:' + aPct + '%"></div>';
                html += '</div>';
                html += '</div>';
            }

            content.innerHTML = html;
        }

        /* ─── Relationships ─── */
        renderRelationships(data) {
            var section = document.getElementById('wf-insp-relationships');
            var tabsEl = document.getElementById('wf-insp-rel-tabs');
            var contentEl = document.getElementById('wf-insp-rel-content');

            if (!data.relationships || Object.keys(data.relationships).length === 0) {
                section.classList.add('is-hidden');
                return;
            }

            section.classList.remove('is-hidden');
            this._relData = data.relationships;

            // Build tabs
            var types = Object.keys(data.relationships);
            var tabsHtml = '';
            types.forEach(function (type, i) {
                var count = data.relationships[type].length;
                var activeClass = i === 0 ? ' is-active' : '';
                tabsHtml += '<button class="wf-rel-tab' + activeClass + '" data-dim="' +
                    this._esc(type) + '" type="button">' +
                    this._capitalize(type) + ' <small>(' + count + ')</small></button>';
            }.bind(this));

            tabsEl.innerHTML = tabsHtml;

            // Wire tab clicks
            var self = this;
            tabsEl.querySelectorAll('.wf-rel-tab').forEach(function (btn) {
                btn.addEventListener('click', function () {
                    tabsEl.querySelectorAll('.wf-rel-tab').forEach(function (b) { b.classList.remove('is-active'); });
                    btn.classList.add('is-active');
                    self._renderRelList(btn.dataset.dim, contentEl);
                });
            });

            // Render first tab
            if (types.length > 0) {
                this._renderRelList(types[0], contentEl);
            }
        }

        _renderRelList(type, contentEl) {
            var items = this._relData[type] || [];
            if (items.length === 0) {
                contentEl.innerHTML = '<div class="wf-rel-empty">No ' + this._esc(type) + ' relationships</div>';
                return;
            }

            var html = '<div class="wf-rel-list">';
            var self = this;

            items.forEach(function (item) {
                var langBadge = '';
                if (type === 'translation' && item.language) {
                    langBadge = '<span class="wf-rel-pill__lang">' + self._esc(item.language) + '</span>';
                }
                var confBadge = '';
                if (item.confidence != null && item.confidence < 1.0) {
                    confBadge = '<span class="wf-rel-pill__confidence">' + item.confidence.toFixed(1) + '</span>';
                }

                html += '<span class="wf-rel-pill" data-term="' + self._esc(item.term) + '" role="button" tabindex="0">' +
                    self._esc(item.term) + langBadge + confBadge + '</span>';
            });

            html += '</div>';
            contentEl.innerHTML = html;

            // Wire pill clicks
            contentEl.querySelectorAll('.wf-rel-pill').forEach(function (pill) {
                pill.addEventListener('click', function () {
                    var term = pill.dataset.term;
                    if (term) self.navigateToTerm(term);
                });
                pill.addEventListener('keydown', function (e) {
                    if (e.key === 'Enter') {
                        var term = pill.dataset.term;
                        if (term) self.navigateToTerm(term);
                    }
                });
            });
        }

        /* ─── Graphemes ─── */
        renderGraphemes(data) {
            var section = document.getElementById('wf-insp-graphemes');
            var wrap = document.getElementById('wf-insp-grapheme-table-wrap');

            if (!data.graphemes || data.graphemes.length === 0) {
                section.classList.add('is-hidden');
                return;
            }

            section.classList.remove('is-hidden');

            var html = '<table class="wf-grapheme-table">';
            html += '<thead><tr><th>#</th><th>Char</th><th>Unicode Name</th></tr></thead>';
            html += '<tbody>';

            data.graphemes.forEach(function (g) {
                var names = (g.unicode_names || []).join(', ');
                html += '<tr>' +
                    '<td>' + g.position + '</td>' +
                    '<td>' + this._esc(g.text) + '</td>' +
                    '<td>' + this._esc(names) + '</td>' +
                '</tr>';
            }.bind(this));

            html += '</tbody></table>';
            wrap.innerHTML = html;
        }

        /* ─── Usage Examples ─── */
        renderExamples(data) {
            var section = document.getElementById('wf-insp-examples');
            var list = document.getElementById('wf-insp-examples-list');

            if (!data.usage_examples || data.usage_examples.length === 0) {
                section.classList.add('is-hidden');
                return;
            }

            section.classList.remove('is-hidden');

            var html = '';
            data.usage_examples.forEach(function (ex) {
                html += '<div class="wf-example-item">"' + this._esc(ex) + '"</div>';
            }.bind(this));

            list.innerHTML = html;
        }

        /* ─── Navigation ─── */
        navigateToTerm(term) {
            if (this._navigateCb) {
                this._navigateCb(term);
            }
            // Also re-open inspector for the new term
            this.open(term);
        }

        /* ─── Helpers ─── */

        _showLoading(show) {
            if (this.loadingEl) {
                this.loadingEl.classList.toggle('is-visible', show);
            }
        }

        _esc(str) {
            if (!str) return '';
            var d = document.createElement('div');
            d.textContent = String(str);
            return d.innerHTML;
        }

        _capitalize(str) {
            if (!str) return '';
            return str.charAt(0).toUpperCase() + str.slice(1);
        }
    }

    // Export
    window.Inspector = Inspector;
})();
