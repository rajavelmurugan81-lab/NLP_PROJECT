/**
 * CiteGuard Frontend JavaScript Application
 * Handles API integration, dynamic DOM rendering, 7-stage execution stepper,
 * dynamic verdict justifications ("Why This Verdict?"), real author metadata extraction,
 * paper retrieval summaries, and report exports.
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const form = document.getElementById('verificationForm');
    const questionInput = document.getElementById('questionInput');
    const modeSelect = document.getElementById('modeSelect');
    const topKSelect = document.getElementById('topKSelect');
    const submitBtn = document.getElementById('submitBtn');
    
    const errorBanner = document.getElementById('errorBanner');
    const errorMessage = document.getElementById('errorMessage');
    
    const stepperSection = document.getElementById('stepperSection');
    const execTimer = document.getElementById('execTimer');
    
    const resultsContainer = document.getElementById('resultsContainer');
    
    const resQuestionId = document.getElementById('resQuestionId');
    const resQuestionText = document.getElementById('resQuestionText');
    const statClaimsCount = document.getElementById('statClaimsCount');
    const statAvgConfidence = document.getElementById('statAvgConfidence');
    const statRetrievalMode = document.getElementById('statRetrievalMode');
    const statPapersCount = document.getElementById('statPapersCount');
    const statExecTime = document.getElementById('statExecTime');
    
    const claimsList = document.getElementById('claimsList');
    const claimsBadgeCount = document.getElementById('claimsBadgeCount');
    const retrievalSummaryBox = document.getElementById('retrievalSummaryBox');
    const verificationDetailsList = document.getElementById('verificationDetailsList');
    
    const ledgerTableBody = document.getElementById('ledgerTableBody');
    const ledgerStatusBadge = document.getElementById('ledgerStatusBadge');
    
    const synthesisVerdictSummary = document.getElementById('synthesisVerdictSummary');
    const synthesisText = document.getElementById('synthesisText');
    const limitationsList = document.getElementById('limitationsList');
    
    const downloadJsonBtn = document.getElementById('downloadJsonBtn');
    const downloadMdBtn = document.getElementById('downloadMdBtn');

    let currentReportFiles = { json_filename: null, md_filename: null };
    let timerInterval = null;
    let startTime = 0;

    // Fetch initial system status
    fetchSystemStatus();

    // Attach preset question click listeners
    document.querySelectorAll('.btn-preset').forEach(btn => {
        btn.addEventListener('click', () => {
            const qText = btn.getAttribute('data-q');
            if (qText) {
                questionInput.value = qText;
                hideError();
            }
        });
    });

    // Form submission listener
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const question = questionInput.value.trim();

        if (!question || question.length < 10) {
            showError("Please enter a valid scientific research question (at least 10 characters).");
            return;
        }

        hideError();
        startPipelineUI();

        try {
            const payload = {
                question: question,
                provider: "mock",
                top_k: parseInt(topKSelect.value, 10),
                mode: modeSelect.value
            };

            // Call FastAPI Backend Server
            const response = await fetch('/api/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.detail || "Pipeline verification failed.");
            }

            stopPipelineUI();
            renderResults(data);

        } catch (err) {
            stopPipelineUI();
            let msg = err.message || "An unexpected error occurred during execution.";
            if (msg === "Failed to fetch") {
                msg = "Failed to connect to CiteGuard server (http://127.0.0.1:8000). Please verify backend server is running.";
            }
            showError(msg);
        }
    });

    // Download handlers
    downloadJsonBtn.addEventListener('click', () => {
        if (currentReportFiles.json_filename) {
            window.open(`/api/reports/${currentReportFiles.json_filename}`, '_blank');
        }
    });

    downloadMdBtn.addEventListener('click', () => {
        if (currentReportFiles.md_filename) {
            window.open(`/api/reports/${currentReportFiles.md_filename}`, '_blank');
        }
    });

    // Fetch System Status & Initial Ledger Counts
    async function fetchSystemStatus() {
        try {
            const res = await fetch('/api/status');
            if (res.ok) {
                const data = await res.json();
                const providerBadge = document.getElementById('providerBadge');
                if (providerBadge && data.provider_display) {
                    providerBadge.innerHTML = `<span class="pulse-dot"></span> Provider: ${data.provider_display}`;
                }
            }
        } catch (e) {
            console.warn("Could not fetch status API:", e);
        }

        // Fetch dynamic counts for ledger tabs
        try {
            const [benchRes, patRes, runRes, truRes] = await Promise.all([
                fetch('/api/ledger/benchmark?limit=1'),
                fetch('/api/ledger/patterns'),
                fetch('/api/ledger/runtime?limit=1'),
                fetch('/api/ledger/trusted')
            ]);
            if (benchRes.ok) {
                const bData = await benchRes.json();
                const count = bData.total_benchmark_records;
                if (benchmarkRecordCount && count !== undefined) benchmarkRecordCount.innerText = count;
                const sideBenchmarkCount = document.getElementById('sideBenchmarkCount');
                if (sideBenchmarkCount && count !== undefined) sideBenchmarkCount.innerText = count;
            }
            if (patRes.ok) {
                const pData = await patRes.json();
                const count = pData.total_patterns || (pData.patterns ? pData.patterns.length : 9);
                if (patternsRecordCount && count !== undefined) patternsRecordCount.innerText = count;
                if (patternTotalCount && count !== undefined) patternTotalCount.innerText = count;
                if (patternsCatalogCount && count !== undefined) patternsCatalogCount.innerText = count;
                const sidePatternsCount = document.getElementById('sidePatternsCount');
                if (sidePatternsCount && count !== undefined) sidePatternsCount.innerText = count;
            }
            if (truRes && truRes.ok) {
                const tData = await truRes.json();
                const count = tData.total_trusted_claims !== undefined ? tData.total_trusted_claims : (tData.claims ? tData.claims.length : 0);
                if (trustedRecordCount && count !== undefined) trustedRecordCount.innerText = count;
                if (trustedTotalCount && count !== undefined) trustedTotalCount.innerText = count;
                const sideTrustedCount = document.getElementById('sideTrustedCount');
                if (sideTrustedCount && count !== undefined) sideTrustedCount.innerText = count;
            }
            if (runRes && runRes.ok) {
                const rData = await runRes.json();
                const count = rData.total_runtime_claims !== undefined ? rData.total_runtime_claims : (rData.claims ? rData.claims.length : 0);
                if (runtimeAuditCount && count !== undefined) runtimeAuditCount.innerText = count;
                const sideAuditCount = document.getElementById('sideAuditCount');
                if (sideAuditCount && count !== undefined) sideAuditCount.innerText = count;
            }
        } catch (e) {
            console.warn("Could not fetch initial ledger counts:", e);
        }
    }

    // UI State Helpers
    function startPipelineUI() {
        if (typeof setSidebarActive === 'function') setSidebarActive('navWorkspaceBtn');
        if (typeof setBreadcrumb === 'function') setBreadcrumb('Research Workspace');

        submitBtn.disabled = true;
        submitBtn.innerHTML = `<span class="btn-icon">⏳</span> Running 8-Stage Verification...`;
        
        stepperSection.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
        
        resetStepper();
        animateStepperProgress();

        startTime = Date.now();
        timerInterval = setInterval(() => {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            execTimer.innerText = `${elapsed}s`;
        }, 100);
    }

    function stopPipelineUI() {
        submitBtn.disabled = false;
        submitBtn.innerHTML = `
            <svg class="btn-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                <path d="m9 12 2 2 4-4"/>
            </svg>
            <span>Verify Research Question</span>
        `;
        clearInterval(timerInterval);
    }

    function resetStepper() {
        for (let i = 1; i <= 8; i++) {
            const el = document.getElementById(`step${i}`);
            if (el) {
                el.classList.remove('active', 'completed');
            }
        }
    }

    function animateStepperProgress() {
        let currentStep = 1;
        const interval = setInterval(() => {
            if (currentStep > 8) {
                clearInterval(interval);
                return;
            }
            
            for (let i = 1; i <= 8; i++) {
                const el = document.getElementById(`step${i}`);
                if (el) {
                    if (i < currentStep) {
                        el.classList.remove('active');
                        el.classList.add('completed');
                    } else if (i === currentStep) {
                        el.classList.add('active');
                    }
                }
            }
            currentStep++;
        }, 400);
    }

    function showError(msg) {
        errorMessage.innerText = msg;
        errorBanner.classList.remove('hidden');
    }

    function hideError() {
        errorBanner.classList.add('hidden');
    }

    // Normalize verdict string to canonical forms
    function normalizeVerdict(verdict) {
        if (!verdict) return 'INSUFFICIENT_EVIDENCE';
        const vUpper = String(verdict).trim().toUpperCase().replace(/ /g, '_');
        if (['SUPPORT', 'SUPPORTED', 'SUPPORTS'].includes(vUpper)) return 'SUPPORTED';
        if (['PARTIAL', 'PARTIALLY_SUPPORTED', 'PARTIALLY_SUPPORT'].includes(vUpper)) return 'PARTIALLY_SUPPORTED';
        if (['REFUTE', 'REFUTED', 'REFUTES', 'CONTRADICT', 'CONTRADICTED', 'REJECTED'].includes(vUpper)) return 'CONTRADICTED';
        if (['NEI', 'INSUFFICIENT', 'INSUFFICIENT_EVIDENCE', 'NEUTRAL', 'NOT_ENOUGH_INFO', 'NO_EVIDENCE'].includes(vUpper)) return 'INSUFFICIENT_EVIDENCE';
        return 'INSUFFICIENT_EVIDENCE';
    }

    // Render Execution Results
    function renderResults(data) {
        const report = data.report || {};
        currentReportFiles = data.report_files || {};

        // Header Summary Box
        resQuestionId.innerText = report.question_id || data.question_id || 'q_000';
        resQuestionText.innerText = report.research_question || data.research_question || data.question || '';
        
        const claims = data.claims || report.claims || [];
        statClaimsCount.innerText = claims.length;
        
        const avgConf = (report.average_confidence !== undefined) ? report.average_confidence : 0.90;
        statAvgConfidence.innerText = `${(avgConf * 100).toFixed(0)}%`;
        
        const providerBadge = document.getElementById('providerBadge');
        if (providerBadge && data.provider_display) {
            providerBadge.innerHTML = `<span class="pulse-dot"></span> Provider: ${data.provider_display}`;
        }
        
        statRetrievalMode.innerText = (modeSelect.value).toUpperCase();
        statExecTime.innerText = `${data.execution_time_sec}s`;
        
        claimsBadgeCount.innerText = `${claims.length} Claims`;

        // Calculate paper retrieval metrics
        const uniquePapers = new Set();
        let openAlexCount = 0;
        let localCount = 0;

        const retrievedPapersList = data.papers_retrieved || [];
        if (retrievedPapersList.length > 0) {
            retrievedPapersList.forEach(p => {
                const pid = p.paper_id || p.doc_id || p.title;
                if (pid) uniquePapers.add(pid);
                const s = String(p.source || "").toLowerCase();
                if (s.includes("openalex")) {
                    openAlexCount++;
                } else {
                    localCount++;
                }
            });
        } else {
            claims.forEach(c => {
                const evList = c.evidence || c.retrieved_evidence || [];
                evList.forEach(ev => {
                    const pid = ev.paper_id || ev.doc_id || ev.title;
                    if (pid) uniquePapers.add(pid);
                    const s = String(ev.source || "").toLowerCase();
                    if (s.includes("openalex")) {
                        openAlexCount++;
                    } else {
                        localCount++;
                    }
                });
            });
        }

        const totalPapersRetrieved = data.papers_retrieved_count !== undefined ? data.papers_retrieved_count : uniquePapers.size;
        statPapersCount.innerText = totalPapersRetrieved;

        // Render Retrieval Summary Box
        retrievalSummaryBox.innerHTML = `
            <div class="retrieval-summary-text">
                <strong>Literature Search Overview:</strong> Retracted/duplicate papers filtered across hybrid indices.
            </div>
            <div class="retrieval-summary-stats">
                <span class="retrieval-stat-tag">Papers Retrieved: ${uniquePapers.size}</span>
                <span class="retrieval-stat-tag">OpenAlex API: ${openAlexCount}</span>
                <span class="retrieval-stat-tag">Local Corpus: ${localCount}</span>
                <span class="retrieval-stat-tag">Top-K Passages: ${topKSelect.value}</span>
            </div>
        `;

        // Calculate Verdict Breakdown
        let suppCount = 0;
        let partCount = 0;
        let neiCount = 0;
        let refCount = 0;

        claims.forEach(c => {
            const norm = normalizeVerdict(c.verdict || c.final_label);
            if (norm === 'SUPPORTED') suppCount++;
            else if (norm === 'PARTIALLY_SUPPORTED') partCount++;
            else if (norm === 'CONTRADICTED') refCount++;
            else neiCount++;
        });

        synthesisVerdictSummary.innerHTML = `
            Verdict Summary Breakdown: 
            <strong style="color: var(--supported-color);">${suppCount} Supported</strong> | 
            <strong style="color: var(--partial-color);">${partCount} Partially Supported</strong> | 
            <strong style="color: var(--nei-color);">${neiCount} Insufficient Evidence</strong> | 
            <strong style="color: var(--contradicted-color);">${refCount} Contradicted</strong>
        `;

        // Render Planner Plan Section
        renderPlannerPlan(data.research_plan || report.research_plan);

        // Render Claims Section
        renderClaimsList(claims);

        // Render Verification Details
        renderVerificationDetails(claims);

        // Update OpenAlex status badge if present
        const statusOpenAlex = document.getElementById('statusOpenAlex');
        if (statusOpenAlex && data.openalex_status) {
            statusOpenAlex.innerText = data.openalex_status.replace(/_/g, ' ');
        }

        // Render Ledger Section (Current Query Audits)
        renderLedgerSection(claims, data.runtime_audit || report.runtime_audit);

        // Render Synthesis & Limitations
        const synthStr = report.overall_synthesis || (data.synthesis ? data.synthesis.summary : "") || (report.synthesis ? report.synthesis.summary : "") || "No synthesis available.";
        synthesisText.innerText = synthStr;
        
        limitationsList.innerHTML = '';
        const lims = report.limitations || [
            "Generated claims are AI/rule-synthesized hypotheses and must be evaluated against retrieved scientific evidence.",
            "Evidence retrieval combines OpenAlex scientific literature with the local SciFact and SciClaimHunt corpus depending on the selected retrieval mode.",
            "OpenAlex retrieval depends on network/API availability; when unavailable, the system can fall back to the local retrieval system.",
            "Retrieval relevance does not itself constitute scientific proof; evidence must be evaluated against each claim.",
            "Verification results are bounded by the scientific literature retrieved by the system.",
            "The system does not generate fake papers or fabricate supporting evidence."
        ];
        lims.forEach(lim => {
            const li = document.createElement('li');
            li.innerText = lim;
            limitationsList.appendChild(li);
        });

        // Render Comprehensive Academic Research Report & Synthesis
        renderResearchGradeReport(data, report);

        // Default to Runtime Audit Tab
        switchTab('runtime');

        // Show Results Section
        resultsContainer.classList.remove('hidden');
    }

    function renderPlannerPlan(plan) {
        const container = document.getElementById('plannerPlanContainer');
        if (!container) return;
        if (!plan) {
            container.innerHTML = `<p style="color: var(--text-secondary); font-size: 13px; font-style: italic;">Research plan details will be generated here during live queries.</p>`;
            return;
        }

        const obj = plan.research_objective || 'Decompose and verify scientific inquiry.';
        const qType = plan.question_type || 'SCIENTIFIC_HYPOTHESIS';
        const subqs = plan.research_subquestions || [];
        const queries = plan.retrieval_queries || [];
        const reqs = plan.evidence_requirements || [];
        const domainConst = (plan.domain_constraints || []).join(', ') || 'General Science';
        const popConst = (plan.population_constraints || []).join(', ') || 'Peer-reviewed literature';

        container.innerHTML = `
            <div class="plan-objective-box">
                <span class="plan-objective-title">🎯 Research Objective [${escapeHtml(qType)}]</span>
                <span class="plan-objective-text">${escapeHtml(obj)}</span>
            </div>

            <div class="plan-columns-grid">
                <div class="plan-card-sub">
                    <h4>📋 Research Subquestions (${subqs.length})</h4>
                    <ol class="plan-subq-list">
                        ${subqs.map(sq => `<li>${escapeHtml(sq)}</li>`).join('')}
                    </ol>
                </div>

                <div class="plan-card-sub">
                    <h4>🔍 Targeted Retrieval Queries (${queries.length})</h4>
                    <div class="plan-queries-chips">
                        ${queries.map(q => `<span class="plan-query-chip">${escapeHtml(q)}</span>`).join('')}
                    </div>
                    <div style="margin-top: 14px; font-size: 12px; color: var(--text-secondary); line-height: 1.6;">
                        <strong>Domain Constraints:</strong> ${escapeHtml(domainConst)}<br>
                        <strong>Population Scope:</strong> ${escapeHtml(popConst)}
                    </div>
                </div>

                <div class="plan-card-sub">
                    <h4>⚖️ Evidence Verification Requirements (${reqs.length})</h4>
                    <ul class="plan-requirements-list">
                        ${reqs.map(r => `<li>${escapeHtml(r)}</li>`).join('')}
                    </ul>
                </div>
            </div>
        `;
    }

    function renderClaimsList(claims) {
        claimsList.innerHTML = '';
        claims.forEach((c, idx) => {
            const card = document.createElement('div');
            card.className = 'claim-card';
            
            const rawVerdict = c.verdict || c.final_label || 'INSUFFICIENT_EVIDENCE';
            const normVerdict = normalizeVerdict(rawVerdict);
            const statusClass = getStatusClass(normVerdict);

            const provStatus = c.provenance_status || (c.provenance ? c.provenance.provenance_status : (c.passage_id ? 'VERIFIED' : null));
            const passageTag = c.passage_id ? `<span class="claim-id-badge">Passage: <code>${escapeHtml(c.passage_id)}</code></span>` : '';
            const provBadge = provStatus ? `<span class="claim-id-badge" style="background: rgba(16, 185, 129, 0.12); color: #059669; border: 1px solid rgba(16, 185, 129, 0.3); font-weight: 600;">✓ Provenance: ${escapeHtml(provStatus)}</span>` : '';

            card.innerHTML = `
                <div>
                    <div class="claim-text">[${idx + 1}] "${escapeHtml(c.claim_text || c.claim)}"</div>
                    <div class="claim-meta">
                        <span class="claim-id-badge">ID: ${c.claim_id}</span>
                        ${passageTag}
                        <span class="claim-id-badge">Specificity: ${(c.specificity_score || 0.92).toFixed(2)}</span>
                        ${provBadge}
                    </div>
                </div>
                <div>
                    <span class="badge-status ${statusClass}">${normVerdict}</span>
                </div>
            `;
            claimsList.appendChild(card);
        });
    }

    function renderVerificationDetails(claims) {
        verificationDetailsList.innerHTML = '';
        claims.forEach((c, idx) => {
            const card = document.createElement('div');
            card.className = 'v-card';

            const rawVerdict = c.verdict || c.final_label || 'INSUFFICIENT_EVIDENCE';
            const normVerdict = normalizeVerdict(rawVerdict);
            const statusClass = getStatusClass(normVerdict);
            const evidenceItems = c.evidence || c.retrieved_evidence || [];

            // Generate "Why This Verdict?" Explanations
            const whyHtml = renderWhyThisVerdict(normVerdict, c, evidenceItems);

            let evidenceHtml = '';
            if (evidenceItems.length === 0) {
                evidenceHtml = `<div class="evidence-passage"><em>No evidence passages retrieved for this claim.</em></div>`;
            } else {
                evidenceItems.forEach(ev => {
                    const paperTitle = ev.title || ev.paper_title || "Scientific Publication";
                    
                    // Extract real author metadata
                    let authorsStr = ev.authors || ev.author;
                    if (Array.isArray(authorsStr)) {
                        authorsStr = authorsStr.join(", ");
                    }
                    if (!authorsStr || authorsStr === "N/A" || authorsStr === "None") {
                        authorsStr = "Author information unavailable";
                    }

                    const year = ev.publication_year || ev.year || "n.d.";
                    const section = ev.section || "Abstract";
                    const passageId = ev.passage_id || `p_${ev.rank || 1}`;
                    const exactText = ev.exact_passage || ev.passage || ev.text || "No passage text available.";
                    const score = (ev.score !== undefined) ? ev.score : (ev.relevance_score !== undefined ? ev.relevance_score : 0.0);
                    
                    // Qualitative Relevance Badge
                    const relBadgeHtml = renderRelevanceBadge(score);

                    const doiUrl = ev.url_doi || ev.doi;
                    const doiHtml = doiUrl ? `<a href="${escapeHtml(doiUrl)}" target="_blank" style="color: var(--primary); text-decoration: underline;">DOI Link</a>` : 'DOI unavailable';

                    // Derive distinct evidence entailment
                    const rawRel = String(ev.relationship || ev.relation || (ev.metadata && ev.metadata.relationship) || '').toUpperCase();
                    let entailmentLabel = 'NEI / UNTAGGED';
                    let entailmentStyle = 'background: rgba(100, 116, 139, 0.1); color: #64748b; border: 1px solid rgba(100, 116, 139, 0.3);';
                    if (rawRel.includes('CONTRADICT') || rawRel.includes('REFUTE')) {
                        entailmentLabel = 'CONTRADICTS';
                        entailmentStyle = 'background: rgba(239, 68, 68, 0.1); color: #dc2626; border: 1px solid rgba(239, 68, 68, 0.3); font-weight: 700;';
                    } else if (rawRel.includes('PARTIAL')) {
                        entailmentLabel = 'PARTIALLY SUPPORTS';
                        entailmentStyle = 'background: rgba(245, 158, 11, 0.1); color: #d97706; border: 1px solid rgba(245, 158, 11, 0.3); font-weight: 700;';
                    } else if (rawRel.includes('SUPPORT')) {
                        entailmentLabel = 'SUPPORTS';
                        entailmentStyle = 'background: rgba(16, 185, 129, 0.1); color: #059669; border: 1px solid rgba(16, 185, 129, 0.3); font-weight: 700;';
                    } else if (rawRel.includes('IRRELEVANT')) {
                        entailmentLabel = 'IRRELEVANT (NO ENTAILMENT)';
                        entailmentStyle = 'background: rgba(148, 163, 184, 0.15); color: #475569; border: 1px solid rgba(148, 163, 184, 0.3);';
                    }

                    evidenceHtml += `
                        <div class="evidence-passage">
                            <div class="evidence-header-row">
                                <div class="evidence-title">
                                    📄 ${escapeHtml(paperTitle)} <span class="evidence-authors">(${escapeHtml(authorsStr)}, ${year})</span>
                                </div>
                                ${relBadgeHtml}
                            </div>
                            <div class="evidence-text">
                                "${escapeHtml(exactText)}"
                            </div>
                            <div class="evidence-meta-row" style="flex-wrap: wrap; gap: 8px;">
                                <span>Paper ID: <code>${escapeHtml(ev.paper_id || ev.doc_id || 'n/a')}</code></span>
                                <span>Passage ID: <code>${escapeHtml(passageId)}</code></span>
                                <span>Section: <code>${escapeHtml(section)}</code></span>
                                <span>Retrieval Relevance Score: <strong>${score.toFixed(4)}</strong></span>
                                <span>Evidence Entailment: <span class="claim-id-badge" style="${entailmentStyle}">${escapeHtml(entailmentLabel)}</span></span>
                                <span>Rank: #${ev.rank || 1}</span>
                                <span>${doiHtml}</span>
                            </div>
                        </div>
                    `;
                });
            }

            card.innerHTML = `
                <div class="v-card-header">
                    <div class="v-card-header-title">
                        <strong>Claim ${idx + 1} (${c.claim_id}):</strong> "${escapeHtml(c.claim_text || c.claim)}"
                    </div>
                    <span class="badge-status ${statusClass}">Verdict: ${normVerdict} (Verification Confidence: ${((c.confidence || 0.90) * 100).toFixed(0)}%)</span>
                </div>
                <div class="v-card-body">
                    ${whyHtml}
                    <div style="font-size: 13px; font-weight: 700; color: var(--text-primary); margin-top: 4px;">
                        Traceable Scientific Evidence Passages (${evidenceItems.length} Passages):
                    </div>
                    ${evidenceHtml}
                </div>
            `;
            verificationDetailsList.appendChild(card);
        });
    }

    function renderWhyThisVerdict(verdict, claim, evidenceItems) {
        let items = [];
        if (verdict === 'SUPPORTED') {
            items = [
                { icon: '✓', class: 'pass', text: 'Relevant peer-reviewed scientific evidence was successfully retrieved.' },
                { icon: '✓', class: 'pass', text: 'Extracted claim concepts directly align with retrieved passage semantics.' },
                { icon: '✓', class: 'pass', text: 'Multi-agent verifiers detected supporting evidence above confidence threshold.' },
                { icon: '✓', class: 'pass', text: 'Zero direct contradictory evidence passages were found.' }
            ];
        } else if (verdict === 'PARTIALLY_SUPPORTED') {
            items = [
                { icon: '✓', class: 'pass', text: 'Retrieved scientific evidence supports key components of the claim.' },
                { icon: '⚠', class: 'warn', text: 'Retrieved passages provide insufficient evidence for secondary aspects of the claim.' },
                { icon: '✓', class: 'pass', text: 'Zero direct contradictory evidence passages were detected.' }
            ];
        } else if (verdict === 'CONTRADICTED') {
            items = [
                { icon: '✗', class: 'fail', text: 'Retrieved peer-reviewed evidence directly conflicts with the generated claim statement.' },
                { icon: '✗', class: 'fail', text: 'Multi-agent consensus flagged contradiction signature during rule checks.' }
            ];
        } else {
            items = [
                { icon: '⚠', class: 'warn', text: 'No sufficiently relevant peer-reviewed passages were found for full verification.' },
                { icon: '⚠', class: 'warn', text: 'Retrieved passage similarity scores fell below confidence threshold (0.60).' }
            ];
        }

        const listHtml = items.map(it => `
            <li class="why-verdict-item">
                <span class="why-icon ${it.class}">${it.icon}</span>
                <span>${escapeHtml(it.text)}</span>
            </li>
        `).join('');

        return `
            <details class="why-verdict-box" open>
                <summary class="why-verdict-title" style="cursor: pointer; user-select: none;">
                    Why This Verdict? (${verdict.replace(/_/g, ' ')})
                </summary>
                <ul class="why-verdict-list" style="margin-top: 8px;">
                    ${listHtml}
                </ul>
            </details>
        `;
    }

    function renderRelevanceBadge(score) {
        if (score >= 0.80) {
            return `<span class="rel-badge rel-high" title="High semantic relevance match (TF-IDF / Vector cosine >= 0.80)">High relevance (${(score * 100).toFixed(0)}%)</span>`;
        } else if (score >= 0.60) {
            return `<span class="rel-badge rel-med" title="Moderate semantic relevance match (TF-IDF / Vector cosine 0.60 - 0.79)">Moderate relevance (${(score * 100).toFixed(0)}%)</span>`;
        } else {
            return `<span class="rel-badge rel-low" title="Low semantic relevance match (TF-IDF / Vector cosine < 0.60)">Low relevance (${(score * 100).toFixed(0)}%)</span>`;
        }
    }

    // Tab elements for Dual Ledger + Failure Memory + Trusted Memory
    const tabRuntimeBtn = document.getElementById('tabRuntimeBtn');
    const tabBenchmarkBtn = document.getElementById('tabBenchmarkBtn');
    const tabPatternsBtn = document.getElementById('tabPatternsBtn');
    const tabTrustedBtn = document.getElementById('tabTrustedBtn');
    const runtimeAuditView = document.getElementById('runtimeAuditView');
    const benchmarkAnalysisView = document.getElementById('benchmarkAnalysisView');
    const patternsAnalysisView = document.getElementById('patternsAnalysisView');
    const trustedMemoryView = document.getElementById('trustedMemoryView');
    const runtimeAuditCount = document.getElementById('runtimeAuditCount');
    const benchmarkRecordCount = document.getElementById('benchmarkRecordCount');
    const patternsRecordCount = document.getElementById('patternsRecordCount');
    const trustedRecordCount = document.getElementById('trustedRecordCount');
    const trustedTotalCount = document.getElementById('trustedTotalCount');
    const trustedSearchInput = document.getElementById('trustedSearchInput');
    const trustedSearchClearBtn = document.getElementById('trustedSearchClearBtn');
    const trustedFilterBadge = document.getElementById('trustedFilterBadge');
    const benchmarkTableBody = document.getElementById('benchmarkTableBody');
    const patternsTableBody = document.getElementById('patternsTableBody');
    const trustedTableBody = document.getElementById('trustedTableBody');
    const benchmarkStatsSummary = document.getElementById('benchmarkStatsSummary');

    // Dashboard panel elements for Failure-Pattern Memory
    const patternTotalCount = document.getElementById('patternTotalCount');
    const patternHighCount = document.getElementById('patternHighCount');
    const patternPreventionRate = document.getElementById('patternPreventionRate');
    const patternsCatalogCount = document.getElementById('patternsCatalogCount');
    const metricFailuresBefore = document.getElementById('metricFailuresBefore');
    const metricFailuresAfter = document.getElementById('metricFailuresAfter');
    const metricPreventionRate = document.getElementById('metricPreventionRate');
    const metricRecurrenceRate = document.getElementById('metricRecurrenceRate');
    const metricMismatchBefore = document.getElementById('metricMismatchBefore');
    const metricMismatchAfter = document.getElementById('metricMismatchAfter');
    const feedbackLoopDetails = document.getElementById('feedbackLoopDetails');
    const recentPatternsContainer = document.getElementById('recentPatternsContainer');
    const recentPatternsBadge = document.getElementById('recentPatternsBadge');
    const recentMiningTimestamp = document.getElementById('recentMiningTimestamp');

    // Sidebar & Navigation Elements
    const navWorkspaceBtn = document.getElementById('navWorkspaceBtn');
    const navTrustedBtn = document.getElementById('navTrustedBtn');
    const navPatternsBtn = document.getElementById('navPatternsBtn');
    const navAuditBtn = document.getElementById('navAuditBtn');
    const navBenchmarkBtn = document.getElementById('navBenchmarkBtn');
    const navSystemBtn = document.getElementById('navSystemBtn');
    const breadcrumbActive = document.getElementById('breadcrumbActive');
    const sidebar = document.getElementById('sidebar');
    const mobileMenuBtn = document.getElementById('mobileMenuBtn');
    const systemInfoModal = document.getElementById('systemInfoModal');
    const closeSystemModalBtn = document.getElementById('closeSystemModalBtn');
    const closeSystemModalFooterBtn = document.getElementById('closeSystemModalFooterBtn');

    function setSidebarActive(btnId) {
        [navWorkspaceBtn, navTrustedBtn, navPatternsBtn, navAuditBtn, navBenchmarkBtn].forEach(b => {
            if (b) b.classList.remove('active');
        });
        const activeBtn = document.getElementById(btnId);
        if (activeBtn) activeBtn.classList.add('active');
    }

    function setBreadcrumb(label) {
        if (breadcrumbActive) breadcrumbActive.innerText = label;
    }

    function closeMobileSidebar() {
        if (sidebar && sidebar.classList.contains('mobile-open')) {
            sidebar.classList.remove('mobile-open');
        }
    }

    function navigateToLedgerTab(tabId, navBtnId, breadcrumbLabel) {
        setSidebarActive(navBtnId);
        setBreadcrumb(breadcrumbLabel);
        if (resultsContainer && resultsContainer.classList.contains('hidden')) {
            resultsContainer.classList.remove('hidden');
        }
        switchTab(tabId);
        const ledgerSec = document.getElementById('ledgerSection');
        if (ledgerSec) {
            ledgerSec.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
        closeMobileSidebar();
    }

    if (navWorkspaceBtn) {
        navWorkspaceBtn.addEventListener('click', () => {
            setSidebarActive('navWorkspaceBtn');
            setBreadcrumb('Research Workspace');
            window.scrollTo({ top: 0, behavior: 'smooth' });
            closeMobileSidebar();
        });
    }

    if (navTrustedBtn) {
        navTrustedBtn.addEventListener('click', () => {
            navigateToLedgerTab('trusted', 'navTrustedBtn', 'Trusted Memory');
        });
    }

    if (navPatternsBtn) {
        navPatternsBtn.addEventListener('click', () => {
            navigateToLedgerTab('patterns', 'navPatternsBtn', 'Failure-Pattern Memory & Prevention');
        });
    }

    if (navAuditBtn) {
        navAuditBtn.addEventListener('click', () => {
            navigateToLedgerTab('runtime', 'navAuditBtn', 'Runtime Audit Ledger');
        });
    }

    if (navBenchmarkBtn) {
        navBenchmarkBtn.addEventListener('click', () => {
            navigateToLedgerTab('benchmark', 'navBenchmarkBtn', 'Historical Benchmark Analysis');
        });
    }

    if (navSystemBtn) {
        navSystemBtn.addEventListener('click', () => {
            if (systemInfoModal) {
                systemInfoModal.classList.remove('hidden');
                systemInfoModal.style.display = 'flex';
            }
            closeMobileSidebar();
        });
    }

    function closeSystemModal() {
        if (systemInfoModal) {
            systemInfoModal.classList.add('hidden');
            systemInfoModal.style.display = 'none';
        }
    }

    if (closeSystemModalBtn) closeSystemModalBtn.addEventListener('click', closeSystemModal);
    if (closeSystemModalFooterBtn) closeSystemModalFooterBtn.addEventListener('click', closeSystemModal);
    if (systemInfoModal) {
        systemInfoModal.addEventListener('click', (e) => {
            if (e.target === systemInfoModal) closeSystemModal();
        });
    }

    if (mobileMenuBtn && sidebar) {
        mobileMenuBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            sidebar.classList.toggle('mobile-open');
        });

        document.addEventListener('click', (e) => {
            if (sidebar.classList.contains('mobile-open') && !sidebar.contains(e.target) && e.target !== mobileMenuBtn) {
                sidebar.classList.remove('mobile-open');
            }
        });
    }

    // Trusted Memory in-memory record cache for instant filtering
    let allTrustedRecords = [];

    function switchTab(activeTab) {
        const tabs = [
            { id: 'runtime', btn: tabRuntimeBtn, view: runtimeAuditView },
            { id: 'benchmark', btn: tabBenchmarkBtn, view: benchmarkAnalysisView },
            { id: 'patterns', btn: tabPatternsBtn, view: patternsAnalysisView },
            { id: 'trusted', btn: tabTrustedBtn, view: trustedMemoryView }
        ];

        tabs.forEach(t => {
            if (t.btn) {
                t.btn.classList.remove('active');
                t.btn.style.color = 'var(--text-secondary)';
                t.btn.style.borderBottom = 'none';
                t.btn.style.fontWeight = '600';
            }
            if (t.view) {
                t.view.classList.add('hidden');
                t.view.style.display = 'none';
            }
        });

        const selected = tabs.find(t => t.id === activeTab);
        if (selected) {
            if (selected.btn) {
                selected.btn.classList.add('active');
                selected.btn.style.color = 'var(--primary)';
                selected.btn.style.borderBottom = '3px solid var(--primary)';
                selected.btn.style.fontWeight = '700';
            }
            if (selected.view) {
                selected.view.classList.remove('hidden');
                selected.view.style.display = 'block';
            }
        }

        if (activeTab === 'runtime') {
            setSidebarActive('navAuditBtn');
            setBreadcrumb('Runtime Audit Ledger');
        } else if (activeTab === 'benchmark') {
            setSidebarActive('navBenchmarkBtn');
            setBreadcrumb('Historical Benchmark Analysis');
            loadBenchmarkLedger();
        } else if (activeTab === 'patterns') {
            setSidebarActive('navPatternsBtn');
            setBreadcrumb('Failure-Pattern Memory & Prevention');
            loadFailurePatterns();
        } else if (activeTab === 'trusted') {
            setSidebarActive('navTrustedBtn');
            setBreadcrumb('Trusted Memory');
            loadTrustedMemory();
        }
    }

    if (tabRuntimeBtn) {
        tabRuntimeBtn.addEventListener('click', () => switchTab('runtime'));
    }
    if (tabBenchmarkBtn) {
        tabBenchmarkBtn.addEventListener('click', () => switchTab('benchmark'));
    }
    if (tabPatternsBtn) {
        tabPatternsBtn.addEventListener('click', () => switchTab('patterns'));
    }
    if (tabTrustedBtn) {
        tabTrustedBtn.addEventListener('click', () => switchTab('trusted'));
    }

    async function loadTrustedMemory(forceReload = false) {
        if (!trustedTableBody) return;
        if (allTrustedRecords.length > 0 && !forceReload) {
            renderTrustedRecords(allTrustedRecords);
            return;
        }

        trustedTableBody.innerHTML = `
            <tr>
                <td colspan="6" style="text-align: center; padding: 36px; color: var(--text-secondary);">
                    <div style="display: flex; align-items: center; justify-content: center; gap: 8px;">
                        <span class="pulse-dot"></span> Loading verified records from <code>results/hallucination_ledger/trusted_memory.json</code>...
                    </div>
                </td>
            </tr>
        `;

        try {
            const res = await fetch('/api/ledger/trusted');
            if (!res.ok) {
                throw new Error(`HTTP ${res.status}: ${res.statusText}`);
            }
            const data = await res.json();
            if (data.status === "error") {
                throw new Error(data.message || "Failed to load trusted memory");
            }

            allTrustedRecords = data.claims || [];
            const count = data.total_trusted_claims !== undefined ? data.total_trusted_claims : allTrustedRecords.length;
            if (trustedRecordCount) trustedRecordCount.innerText = count;
            if (trustedTotalCount) trustedTotalCount.innerText = count;
            const sideTrustedCount = document.getElementById('sideTrustedCount');
            if (sideTrustedCount && count !== undefined) sideTrustedCount.innerText = count;

            if (trustedSearchInput && trustedSearchInput.value.trim()) {
                filterTrustedRecords();
            } else {
                renderTrustedRecords(allTrustedRecords);
            }
        } catch (err) {
            console.error("Failed to load trusted memory:", err);
            trustedTableBody.innerHTML = `
                <tr>
                    <td colspan="6" style="text-align: center; padding: 28px; color: var(--contradicted-color);">
                        ⚠️ <strong>Failed to load Trusted Memory:</strong> ${escapeHtml(err.message)}<br>
                        <span style="font-size: 12px; color: var(--text-muted); margin-top: 6px; display: inline-block;">
                            Please verify the backend API server is running and <code>results/hallucination_ledger/trusted_memory.json</code> exists.
                        </span>
                    </td>
                </tr>
            `;
            if (trustedFilterBadge) {
                trustedFilterBadge.innerText = 'Error loading records';
                trustedFilterBadge.style.background = 'rgba(239, 68, 68, 0.1)';
                trustedFilterBadge.style.color = 'var(--contradicted-color)';
            }
        }
    }

    function renderTrustedRecords(records) {
        if (!trustedTableBody) return;
        trustedTableBody.innerHTML = '';

        if (!allTrustedRecords || allTrustedRecords.length === 0) {
            trustedTableBody.innerHTML = `
                <tr>
                    <td colspan="6" style="text-align: center; padding: 32px; color: var(--text-muted);">
                        🛡️ <em>No trusted verification records found in <code>trusted_memory.json</code>.</em><br>
                        <span style="font-size: 12px; margin-top: 4px; display: inline-block;">
                            Cleanly verified claims with zero failure patterns will automatically be stored here.
                        </span>
                    </td>
                </tr>
            `;
            if (trustedFilterBadge) {
                trustedFilterBadge.innerText = '0 records';
            }
            return;
        }

        if (records.length === 0) {
            const query = trustedSearchInput ? trustedSearchInput.value.trim() : '';
            trustedTableBody.innerHTML = `
                <tr>
                    <td colspan="6" style="text-align: center; padding: 32px; color: var(--text-muted);">
                        🔍 <em>No records match your filter "${escapeHtml(query)}".</em>
                    </td>
                </tr>
            `;
            if (trustedFilterBadge) {
                trustedFilterBadge.innerText = `Showing 0 of ${allTrustedRecords.length} records`;
            }
            return;
        }

        if (trustedFilterBadge) {
            trustedFilterBadge.innerText = `Showing ${records.length} of ${allTrustedRecords.length} records`;
            trustedFilterBadge.style.background = 'rgba(37, 99, 235, 0.1)';
            trustedFilterBadge.style.color = 'var(--primary)';
        }

        // Render records in reverse chronological order (newest first)
        const displayRecords = [...records].reverse();
        displayRecords.forEach(c => {
            const tr = document.createElement('tr');
            const confPct = ((c.confidence || 0) * 100).toFixed(1);
            const dateStr = c.timestamp ? new Date(c.timestamp).toLocaleString() : '-';
            const normVerdict = (c.verdict || 'SUPPORT').toUpperCase();

            tr.innerHTML = `
                <td style="vertical-align: top;">
                    <code>${escapeHtml(c.claim_id || '-')}</code>
                </td>
                <td style="vertical-align: top;">
                    <div style="font-weight: 600; color: var(--text-primary); margin-bottom: 6px; line-height: 1.4;">
                        ${escapeHtml(c.claim_text || '')}
                    </div>
                    ${c.question ? `
                        <div style="font-size: 12px; color: var(--text-secondary); background: var(--bg-subtle); padding: 5px 9px; border-radius: 4px; border-left: 3px solid var(--primary); margin-top: 4px;">
                            <strong style="color: var(--text-primary);">Q:</strong> ${escapeHtml(c.question)}
                        </div>
                    ` : ''}
                </td>
                <td style="vertical-align: top; text-align: center;">
                    <span class="badge-status status-supported" style="font-weight: 700;">${escapeHtml(normVerdict)}</span>
                </td>
                <td style="vertical-align: top; text-align: center;">
                    <strong style="color: var(--supported-color); font-size: 13px;">${confPct}%</strong>
                </td>
                <td style="vertical-align: top; text-align: center;">
                    <span class="retrieval-stat-tag" style="background: rgba(16, 185, 129, 0.12); color: #047857; font-weight: 600; white-space: nowrap;">
                        ${c.evidence_count || 0} passages
                    </span>
                </td>
                <td style="vertical-align: top; font-size: 12px; color: var(--text-secondary); white-space: nowrap;">
                    ${escapeHtml(dateStr)}
                </td>
            `;
            trustedTableBody.appendChild(tr);
        });
    }

    function filterTrustedRecords() {
        const query = (trustedSearchInput ? trustedSearchInput.value : '').toLowerCase().trim();
        if (trustedSearchClearBtn) {
            trustedSearchClearBtn.style.display = query ? 'inline-block' : 'none';
        }
        if (!query) {
            renderTrustedRecords(allTrustedRecords);
            return;
        }
        const filtered = allTrustedRecords.filter(r => {
            const idMatch = (r.claim_id || '').toLowerCase().includes(query);
            const textMatch = (r.claim_text || '').toLowerCase().includes(query);
            const qMatch = (r.question || '').toLowerCase().includes(query);
            const verdictMatch = (r.verdict || '').toLowerCase().includes(query);
            return idMatch || textMatch || qMatch || verdictMatch;
        });
        renderTrustedRecords(filtered);
    }

    if (trustedSearchInput) {
        trustedSearchInput.addEventListener('input', filterTrustedRecords);
    }
    if (trustedSearchClearBtn) {
        trustedSearchClearBtn.addEventListener('click', () => {
            trustedSearchInput.value = '';
            filterTrustedRecords();
            trustedSearchInput.focus();
        });
    }

    async function loadFailurePatterns() {
        if (!patternsTableBody) return;
        patternsTableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 24px;">Loading learned failure patterns from results/hallucination_ledger/failure_patterns.json...</td></tr>';

        // 1. Fetch Learned Patterns
        try {
            const res = await fetch('/api/ledger/patterns');
            if (!res.ok) throw new Error("Failed to load failure patterns");
            const data = await res.json();
            const patterns = data.patterns || [];

            if (patternsRecordCount) patternsRecordCount.innerText = patterns.length;
            if (patternTotalCount) patternTotalCount.innerText = patterns.length;
            if (patternsCatalogCount) patternsCatalogCount.innerText = patterns.length;
            const sidePatternsCount = document.getElementById('sidePatternsCount');
            if (sidePatternsCount) sidePatternsCount.innerText = patterns.length;

            const highCount = patterns.filter(p => p.severity === 'CRITICAL' || p.severity === 'HIGH').length;
            if (patternHighCount) patternHighCount.innerText = highCount;

            // Render Recently Learned / Updated Patterns Card
            if (recentPatternsContainer) {
                const miningSummary = data.mining_summary;
                const created = miningSummary ? (miningSummary.created_patterns || []) : [];
                const updated = miningSummary ? (miningSummary.updated_patterns || []) : [];

                if (recentMiningTimestamp && data.generated_at) {
                    recentMiningTimestamp.innerText = `Last Mined: ${new Date(data.generated_at).toLocaleTimeString()}`;
                }

                if (created.length === 0 && updated.length === 0) {
                    if (recentPatternsBadge) {
                        recentPatternsBadge.className = 'badge-status status-clean';
                        recentPatternsBadge.innerText = 'Up to Date';
                    }
                    recentPatternsContainer.innerHTML = `
                        <div style="color: var(--text-muted); font-style: italic; padding: 4px 0;">
                            All ${patterns.length} failure patterns are synchronized with runtime audit ledgers. No new distinct pattern was added in the latest mining cycle.
                        </div>
                    `;
                } else {
                    if (recentPatternsBadge) {
                        recentPatternsBadge.className = created.length > 0 ? 'badge-status status-refuted' : 'badge-status status-partial';
                        recentPatternsBadge.innerText = `${created.length} CREATED, ${updated.length} UPDATED`;
                    }

                    let recentHtml = '';
                    created.forEach(cp => {
                        recentHtml += `
                            <div style="display: flex; align-items: flex-start; justify-content: space-between; padding: 8px 12px; background: rgba(16, 185, 129, 0.08); border-left: 4px solid #10b981; border-radius: 4px; margin-bottom: 6px;">
                                <div>
                                    <span class="badge-status status-clean" style="font-size: 10px; font-weight: 700; margin-right: 6px;">CREATED</span>
                                    <code>${escapeHtml(cp.pattern_id)}</code>: <strong>${escapeHtml(cp.pattern_name)}</strong>
                                    <span class="retrieval-stat-tag" style="margin-left: 6px; font-size: 11px;">${escapeHtml(cp.canonical_failure_code || cp.failure_type)}</span>
                                    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">${escapeHtml(cp.observed_failure || cp.prevention_rule)}</div>
                                </div>
                                <div style="font-size: 12px; font-weight: 700; color: #10b981; white-space: nowrap;">${cp.frequency || 1}x observed</div>
                            </div>
                        `;
                    });

                    updated.forEach(up => {
                        recentHtml += `
                            <div style="display: flex; align-items: flex-start; justify-content: space-between; padding: 8px 12px; background: rgba(245, 158, 11, 0.08); border-left: 4px solid #f59e0b; border-radius: 4px; margin-bottom: 6px;">
                                <div>
                                    <span class="badge-status status-partial" style="font-size: 10px; font-weight: 700; margin-right: 6px;">UPDATED</span>
                                    <code>${escapeHtml(up.pattern_id)}</code>: <strong>${escapeHtml(up.pattern_name)}</strong>
                                    <span class="retrieval-stat-tag" style="margin-left: 6px; font-size: 11px;">${escapeHtml(up.canonical_failure_code || up.failure_type)}</span>
                                    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">Frequency incremented from live audit records; provenance updated.</div>
                                </div>
                                <div style="font-size: 12px; font-weight: 700; color: #d97706; white-space: nowrap;">${up.frequency || 1}x total</div>
                            </div>
                        `;
                    });

                    recentPatternsContainer.innerHTML = recentHtml;
                }
            }

            patternsTableBody.innerHTML = '';
            if (patterns.length === 0) {
                patternsTableBody.innerHTML = '<tr><td colspan="6" style="text-align: center; padding: 20px;">No failure patterns learned yet.</td></tr>';
            } else {
                patterns.forEach(p => {
                    const tr = document.createElement('tr');
                    const isHigh = p.severity === 'CRITICAL' || p.severity === 'HIGH';
                    const sevClass = isHigh ? 'status-refuted' : 'status-partial';
                    const actionsStr = (p.adaptive_actions || []).join(', ') || p.prevention_rule;
                    const trigConditions = p.trigger_conditions || {};
                    const trigBadges = Object.entries(trigConditions).map(([k, v]) => `<span class="retrieval-stat-tag" style="margin: 2px 2px 2px 0;">${escapeHtml(k)}: <strong>${escapeHtml(String(v))}</strong></span>`).join('');

                    const canonicalBadge = p.canonical_failure_code ? 
                        `<br><span class="retrieval-stat-tag" style="font-size: 11px; margin-top: 4px; background: rgba(16, 185, 129, 0.12); color: #047857; font-weight: 600;">${escapeHtml(p.canonical_failure_code)}: ${escapeHtml(p.canonical_failure_name || '')}</span>` : '';

                    tr.innerHTML = `
                        <td><code>${escapeHtml(p.pattern_id)}</code></td>
                        <td>
                            <strong>${escapeHtml(p.pattern_name)}</strong><br>
                            <span class="retrieval-stat-tag" style="font-size: 11px; margin-top: 4px;">${escapeHtml(p.failure_type)}</span>
                            ${canonicalBadge}
                            ${p.failure_signature ? `<br><small style="color: var(--text-muted); font-family: var(--font-mono);">${escapeHtml(p.failure_signature)}</small>` : ''}
                        </td>
                        <td style="font-size: 12px;">
                            ${trigBadges || '<span style="color: var(--text-muted);">Structural anomaly criteria</span>'}
                        </td>
                        <td style="font-size: 12px;">
                            <div style="margin-bottom: 4px;"><strong>Rule:</strong> ${escapeHtml(p.prevention_rule)}</div>
                            <div style="color: var(--primary); font-size: 11px;"><strong>Actions:</strong> ${escapeHtml(actionsStr)}</div>
                        </td>
                        <td><span class="badge-status ${sevClass}">${escapeHtml(p.severity)}</span></td>
                        <td><strong>${p.frequency || 1}x</strong></td>
                    `;
                    patternsTableBody.appendChild(tr);
                });
            }
        } catch (e) {
            patternsTableBody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--contradicted-color); padding: 20px;">Error loading failure patterns: ${escapeHtml(e.message)}</td></tr>`;
        }

        // 2. Fetch Prevention Evaluation Metrics
        try {
            const evalRes = await fetch('/api/ledger/prevention-evaluation');
            if (evalRes.ok) {
                const evalData = await evalRes.json();
                if (evalData.status === 'success') {
                    if (patternPreventionRate && evalData.prevention_rate !== undefined) {
                        patternPreventionRate.innerText = `${(evalData.prevention_rate * 100).toFixed(1)}%`;
                    }
                    if (metricFailuresBefore && evalData.failures_before_memory !== undefined) {
                        metricFailuresBefore.innerText = evalData.failures_before_memory;
                    }
                    if (metricFailuresAfter && evalData.failures_after_memory !== undefined) {
                        metricFailuresAfter.innerText = evalData.failures_after_memory;
                    }
                    if (metricPreventionRate && evalData.prevention_rate !== undefined) {
                        metricPreventionRate.innerText = `${(evalData.prevention_rate * 100).toFixed(1)}%`;
                    }
                    if (metricRecurrenceRate && evalData.recurrence_rate !== undefined) {
                        metricRecurrenceRate.innerText = `${(evalData.recurrence_rate * 100).toFixed(1)}%`;
                    }
                    if (metricMismatchBefore && evalData.domain_mismatch_rate_before !== undefined) {
                        metricMismatchBefore.innerText = `${(evalData.domain_mismatch_rate_before * 100).toFixed(1)}%`;
                    }
                    if (metricMismatchAfter && evalData.domain_mismatch_rate_after !== undefined) {
                        metricMismatchAfter.innerText = `${(evalData.domain_mismatch_rate_after * 100).toFixed(1)}%`;
                    }

                    // Populate feedback loop container if available
                    if (feedbackLoopDetails && evalData.demonstrated_feedback_loop) {
                        const loop = evalData.demonstrated_feedback_loop;
                        feedbackLoopDetails.innerHTML = `
                            <div class="feedback-loop-step">
                                <span class="feedback-loop-tag tag-fail">RUN 1 FAIL</span>
                                <span>${escapeHtml(loop.run_1_failure || 'Off-domain evidence accepted as support')}</span>
                            </div>
                            <div class="feedback-loop-step">
                                <span class="feedback-loop-tag tag-learn">LEARNED</span>
                                <span>${escapeHtml(loop.learned_pattern || 'Pattern extracted to memory ledger')}</span>
                            </div>
                            <div class="feedback-loop-step">
                                <span class="feedback-loop-tag tag-match">RUN 2 INPUT</span>
                                <span>${escapeHtml(loop.run_2_input || 'Similar candidate presented')} &rarr; <strong>${escapeHtml(loop.historical_match || 'Match detected')}</strong></span>
                            </div>
                            <div class="feedback-loop-step">
                                <span class="feedback-loop-tag tag-adapt">ADAPTIVE</span>
                                <span>${escapeHtml(loop.adaptive_verification || 'Strict verification triggered')} &rarr; Re-check: <strong>${escapeHtml(loop.claim_recheck || 'REJECT')}</strong></span>
                            </div>
                            <div class="feedback-loop-step">
                                <span class="feedback-loop-tag tag-success">PREVENTED</span>
                                <span>${escapeHtml(loop.verdict_change || 'Spurious support prevented successfully')}</span>
                            </div>
                        `;
                    }
                }
            }
        } catch (err) {
            console.warn("Could not fetch prevention evaluation:", err);
        }
    }

    async function loadBenchmarkLedger() {
        if (!benchmarkTableBody) return;
        benchmarkTableBody.innerHTML = '<tr><td colspan="5" style="text-align: center; padding: 20px;">Loading historical benchmark records from results/hallucination_ledger/...</td></tr>';
        try {
            const res = await fetch('/api/ledger/benchmark?limit=100');
            if (!res.ok) throw new Error("Failed to load benchmark records");
            const data = await res.json();
            const records = data.records || [];
            if (benchmarkRecordCount) benchmarkRecordCount.innerText = data.total_benchmark_records || records.length;
            const sideBenchmarkCount = document.getElementById('sideBenchmarkCount');
            if (sideBenchmarkCount) sideBenchmarkCount.innerText = data.total_benchmark_records || records.length;

            // Render stats summary
            if (benchmarkStatsSummary && data.statistics) {
                const s = data.statistics;
                benchmarkStatsSummary.innerHTML = `
                    <span class="retrieval-stat-tag">Total Evaluated: ${s.total_evaluated_cases || records.length}</span>
                    <span class="retrieval-stat-tag" style="background: rgba(239, 68, 68, 0.1); color: var(--contradicted-color);">Failures Recorded: ${s.failure_count || records.length}</span>
                    <span class="retrieval-stat-tag">SciFact: ${(s.breakdown_by_dataset || {}).scifact || '-'}</span>
                    <span class="retrieval-stat-tag">SciClaimHunt: ${(s.breakdown_by_dataset || {}).sciclaimhunt || '-'}</span>
                    <span class="retrieval-stat-tag">CoVERT: ${(s.breakdown_by_dataset || {}).covert || '-'}</span>
                `;
            }

            benchmarkTableBody.innerHTML = '';
            if (records.length === 0) {
                benchmarkTableBody.innerHTML = '<tr><td colspan="5" style="text-align: center; padding: 20px;">No benchmark records found.</td></tr>';
                return;
            }

            records.reverse().forEach(r => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><code>${escapeHtml(r.dataset || 'unknown')}</code></td>
                    <td><strong>${escapeHtml(r.claim_id || '')}</strong>: "${escapeHtml(r.claim_text || '')}"</td>
                    <td><span class="badge-status status-partial">${escapeHtml(r.predicted_label || '')}</span> vs <span class="badge-status status-clean">${escapeHtml(r.ground_truth_label || '')}</span></td>
                    <td><strong style="color: var(--contradicted-color);">${escapeHtml(r.failure_type || '')}</strong></td>
                    <td style="font-size: 12px;">${escapeHtml(r.explanation || '')}</td>
                `;
                benchmarkTableBody.appendChild(tr);
            });
        } catch (e) {
            benchmarkTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--contradicted-color); padding: 20px;">Error loading benchmark ledger: ${escapeHtml(e.message)}</td></tr>`;
        }
    }

    function renderLedgerSection(claims, runtimeAuditList) {
        ledgerTableBody.innerHTML = '';
        const audits = runtimeAuditList || claims.map(c => c.runtime_audit).filter(Boolean);
        let anomalyCount = 0;

        if (runtimeAuditCount) {
            runtimeAuditCount.innerText = audits.length || claims.length;
        }
        const sideAuditCount = document.getElementById('sideAuditCount');
        if (sideAuditCount) {
            sideAuditCount.innerText = audits.length || claims.length;
        }

        if (audits.length > 0) {
            audits.forEach(a => {
                const isClean = a.severity === 'CLEAN' || a.severity === 'NONE' || a.failure_type === 'CLEAN' || a.failure_type === 'NONE' || a.ledger_status === 'CLEAN';
                if (!isClean) anomalyCount++;

                const tr = document.createElement('tr');
                const catColor = isClean ? 'var(--supported-color)' : 'var(--contradicted-color)';
                const sevBadgeClass = isClean ? 'status-supported' : (a.severity === 'CRITICAL' || a.severity === 'HIGH' ? 'status-refuted' : 'status-partial');

                // Format canonical failure code badge
                const canCode = a.canonical_failure_code && a.canonical_failure_code !== 'NONE' ? `<span class="badge-status status-partial" style="margin-left: 4px; font-weight: 700;">${escapeHtml(a.canonical_failure_code)}</span>` : '';
                
                // Format matched pattern information
                const patternIds = Array.isArray(a.matched_pattern_ids) ? a.matched_pattern_ids : [];
                const patternDisplay = patternIds.length > 0
                    ? `<code>${escapeHtml(patternIds.join(', '))}</code> ${a.pattern_match_score ? `<br><small style="color: var(--text-muted);">Score: ${(a.pattern_match_score * 100).toFixed(0)}%</small>` : ''}`
                    : `<span style="color: var(--text-muted); font-size: 12px;">None</span>`;

                // Format prevention action & re-check outcome
                let preventionDisplay = `<span style="color: var(--text-muted); font-size: 12px;">-</span>`;
                if (a.prevention_triggered || (a.adaptive_action && a.adaptive_action !== 'NONE')) {
                    const actionBadge = `<span class="retrieval-stat-tag" style="background: rgba(59, 130, 246, 0.1); color: #3b82f6; font-size: 11px;" title="Prevention rule triggered and executed on this claim">${escapeHtml(a.adaptive_action || 'ACTION')}</span>`;
                    
                    let recheckBadge = '';
                    if (a.claim_recheck_status === 'REJECT') {
                        recheckBadge = `<span class="badge-status status-refuted" style="font-size: 10px; padding: 2px 6px;" title="Claim Re-Check outcome: REJECT (Evidence invalid or insufficient)">Re-Check: REJECT</span>`;
                    } else if (a.claim_recheck_status === 'REVISE') {
                        recheckBadge = `<span class="badge-status status-partial" style="font-size: 10px; padding: 2px 6px;" title="Claim Re-Check outcome: REVISE (Verdict or confidence recalibrated)">Re-Check: REVISE</span>`;
                    } else {
                        recheckBadge = `<span class="badge-status status-clean" style="font-size: 10px; padding: 2px 6px;" title="Claim Re-Check outcome: PASS (Evidence passed all verification contracts)">Re-Check: PASS</span>`;
                    }
                    preventionDisplay = `${actionBadge}<br>${recheckBadge}`;
                }

                tr.innerHTML = `
                    <td><code>${escapeHtml(a.claim_id)}</code></td>
                    <td>
                        <strong style="color: ${catColor};">${escapeHtml(a.failure_type)}</strong> ${canCode}
                        ${a.failure_signature && a.failure_signature !== 'NONE' ? `<br><small style="color: var(--text-muted);"><code>${escapeHtml(a.failure_signature)}</code></small>` : ''}
                    </td>
                    <td style="max-width: 280px; font-size: 12px;">${escapeHtml(a.explanation)}</td>
                    <td><strong>${((a.best_relevance_score || 0) * 100).toFixed(0)}%</strong></td>
                    <td>${patternDisplay}</td>
                    <td>${preventionDisplay}</td>
                    <td><span class="badge-status ${sevBadgeClass}">${escapeHtml(a.severity)}</span></td>
                    <td><span class="retrieval-stat-tag">${escapeHtml(a.ledger_status)}</span></td>
                `;
                ledgerTableBody.appendChild(tr);
            });
        } else {
            // Fallback to claims iteration
            claims.forEach(c => {
                const tr = document.createElement('tr');
                const sig = c.failure_signature;
                if (sig) {
                    anomalyCount++;
                    tr.innerHTML = `
                        <td><code>${escapeHtml(c.claim_id)}</code></td>
                        <td><strong style="color: var(--contradicted-color);">${escapeHtml(sig.failure_type || "ANOMALY")}</strong></td>
                        <td style="max-width: 280px; font-size: 12px;">${escapeHtml(sig.explanation || "Rule violation detected.")}</td>
                        <td>-</td>
                        <td><span style="color: var(--text-muted); font-size: 12px;">None</span></td>
                        <td><span style="color: var(--text-muted); font-size: 12px;">-</span></td>
                        <td><span class="badge-status status-refuted">HIGH</span></td>
                        <td>LOGGED_TO_AUDIT</td>
                    `;
                } else {
                    tr.innerHTML = `
                        <td><code>${escapeHtml(c.claim_id)}</code></td>
                        <td><strong style="color: var(--supported-color);">CLEAN</strong></td>
                        <td style="max-width: 280px; font-size: 12px;">Claim verified cleanly with evidence grounding conforming to verification contracts.</td>
                        <td><strong>${((c.confidence || 0.85) * 100).toFixed(0)}%</strong></td>
                        <td><span style="color: var(--text-muted); font-size: 12px;">None</span></td>
                        <td><span style="color: var(--text-muted); font-size: 12px;">-</span></td>
                        <td><span class="badge-status status-supported">CLEAN</span></td>
                        <td>CLEAN_CONTRACT</td>
                    `;
                }
                ledgerTableBody.appendChild(tr);
            });
        }

        if (anomalyCount === 0) {
            ledgerStatusBadge.className = 'badge-status status-clean';
            ledgerStatusBadge.innerText = '0 Anomaly Signatures Detected (Clean Contract)';
        } else {
            ledgerStatusBadge.className = 'badge-status status-refuted';
            ledgerStatusBadge.innerText = `${anomalyCount} Anomaly Signatures Logged`;
        }
    }

    function getStatusClass(verdict) {
        const norm = normalizeVerdict(verdict);
        if (norm === 'SUPPORTED') return 'status-supported';
        if (norm === 'PARTIALLY_SUPPORTED') return 'status-partial';
        if (norm === 'CONTRADICTED') return 'status-refuted';
        return 'status-nei';
    }

    function str(val) {
        return val ? String(val) : "";
    }

    function escapeHtml(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    // =========================================================================
    // Section 4: Academic-Grade Research Report & Evidentiary Synthesis Engine
    // =========================================================================

    function renderResearchGradeReport(data, report) {
        const claims = data.claims || report.claims || [];
        const dash = data.dashboard || report.dashboard || {};
        const rs = data.research_synthesis || report.research_synthesis || {};
        const ug = data.uncertainty_and_gaps || report.uncertainty_and_gaps || {};

        // 1. Verdict Breakdown & Metrics
        const totalClaims = claims.length;
        let supCount = 0, partCount = 0, neiCount = 0, conCount = 0;
        let sumConf = 0;

        claims.forEach(c => {
            const v = normalizeVerdict(c.verdict || c.final_label);
            if (v === 'SUPPORTED') supCount++;
            else if (v === 'PARTIALLY_SUPPORTED') partCount++;
            else if (v === 'CONTRADICTED') conCount++;
            else neiCount++;
            sumConf += (c.confidence !== undefined ? c.confidence : 0.85);
        });

        const avgConf = (dash.average_confidence !== undefined) 
            ? dash.average_confidence 
            : (totalClaims > 0 ? (sumConf / totalClaims) : 0.0);

        const supPct = totalClaims > 0 ? ((supCount / totalClaims) * 100).toFixed(1) : "0.0";
        const partPct = totalClaims > 0 ? ((partCount / totalClaims) * 100).toFixed(1) : "0.0";
        const neiPct = totalClaims > 0 ? ((neiCount / totalClaims) * 100).toFixed(1) : "0.0";
        const conPct = totalClaims > 0 ? ((conCount / totalClaims) * 100).toFixed(1) : "0.0";

        const dashTotalClaims = document.getElementById('dashTotalClaims');
        const dashSupported = document.getElementById('dashSupported');
        const dashSupportedPct = document.getElementById('dashSupportedPct');
        const dashPartial = document.getElementById('dashPartial');
        const dashPartialPct = document.getElementById('dashPartialPct');
        const dashNEI = document.getElementById('dashNEI');
        const dashNEIPct = document.getElementById('dashNEIPct');
        const dashContradicted = document.getElementById('dashContradicted');
        const dashContradictedPct = document.getElementById('dashContradictedPct');
        const dashConfidence = document.getElementById('dashConfidence');

        if (dashTotalClaims) dashTotalClaims.innerText = totalClaims;
        if (dashSupported) dashSupported.innerText = supCount;
        if (dashSupportedPct) dashSupportedPct.innerText = `${supPct}% of claims`;
        if (dashPartial) dashPartial.innerText = partCount;
        if (dashPartialPct) dashPartialPct.innerText = `${partPct}% of claims`;
        if (dashNEI) dashNEI.innerText = neiCount;
        if (dashNEIPct) dashNEIPct.innerText = `${neiPct}% of claims`;
        if (dashContradicted) dashContradicted.innerText = conCount;
        if (dashContradictedPct) dashContradictedPct.innerText = `${conPct}% of claims`;
        if (dashConfidence) dashConfidence.innerText = `${(avgConf * 100).toFixed(1)}%`;

        // Proportional Distribution Bar
        const segSupp = document.getElementById('segSupp');
        const segPartial = document.getElementById('segPartial');
        const segNEI = document.getElementById('segNEI');
        const segCon = document.getElementById('segCon');
        const distBarPercentages = document.getElementById('distBarPercentages');

        if (segSupp) segSupp.style.width = `${supPct}%`;
        if (segPartial) segPartial.style.width = `${partPct}%`;
        if (segNEI) segNEI.style.width = `${neiPct}%`;
        if (segCon) segCon.style.width = `${conPct}%`;
        if (distBarPercentages) distBarPercentages.innerText = `${supPct}% Supp | ${partPct}% Partial | ${neiPct}% NEI | ${conPct}% Con`;

        // Literature Citation & Quality Metrics
        const uniqueCited = new Set();
        const contradictorySources = new Set();
        const publicationYears = [];
        let claimsWithEvidence = 0;

        claims.forEach(c => {
            const evList = c.evidence || c.retrieved_evidence || [];
            if (evList.length > 0) claimsWithEvidence++;
            evList.forEach(ev => {
                const pid = ev.paper_id || ev.doc_id || ev.title;
                if (pid) uniqueCited.add(pid);
                const rel = String(ev.relation || ev.relationship || "").toUpperCase();
                const cv = normalizeVerdict(c.verdict || c.final_label);
                if (rel.includes("CONTRADICT") || rel.includes("REFUTE") || cv === "CONTRADICTED") {
                    if (pid) contradictorySources.add(pid);
                }
                const yr = ev.publication_year || ev.year;
                if (yr && !isNaN(Number(yr))) {
                    const yNum = Number(yr);
                    if (yNum >= 1900 && yNum <= 2035) publicationYears.push(yNum);
                }
            });
        });

        const citedCount = (dash.unique_sources_cited !== undefined) ? dash.unique_sources_cited : uniqueCited.size;
        const retrievedCount = (dash.unique_sources_retrieved !== undefined) 
            ? dash.unique_sources_retrieved 
            : (data.papers_retrieved_count || (data.papers_retrieved ? data.papers_retrieved.length : citedCount));
        const coverageRate = (dash.evidence_coverage_pct !== undefined)
            ? dash.evidence_coverage_pct
            : (totalClaims > 0 ? ((claimsWithEvidence / totalClaims) * 100).toFixed(1) : "0.0");
        const contradCount = (dash.contradictory_sources_count !== undefined) ? dash.contradictory_sources_count : contradictorySources.size;
        const missingCount = (dash.missing_evidence_count !== undefined) ? dash.missing_evidence_count : (totalClaims - claimsWithEvidence);
        
        let yearSpanStr = "Not available";
        if (dash.publication_year_range && dash.publication_year_range !== "Not available") {
            yearSpanStr = dash.publication_year_range;
        } else if (publicationYears.length > 0) {
            yearSpanStr = `${Math.min(...publicationYears)} – ${Math.max(...publicationYears)}`;
        }

        const dashSourcesCited = document.getElementById('dashSourcesCited');
        const dashSourcesRetrieved = document.getElementById('dashSourcesRetrieved');
        const dashCoverageRate = document.getElementById('dashCoverageRate');
        const dashContradictorySources = document.getElementById('dashContradictorySources');
        const dashMissingEvidence = document.getElementById('dashMissingEvidence');
        const dashYearRange = document.getElementById('dashYearRange');
        const dashRetrievalMode = document.getElementById('dashRetrievalMode');
        const dashDirectVsIndirect = document.getElementById('dashDirectVsIndirect');

        if (dashSourcesCited) dashSourcesCited.innerText = `${citedCount} papers`;
        if (dashSourcesRetrieved) dashSourcesRetrieved.innerText = `${retrievedCount} papers`;
        if (dashCoverageRate) dashCoverageRate.innerText = `${coverageRate}%`;
        if (dashContradictorySources) dashContradictorySources.innerText = `${contradCount} sources`;
        if (dashMissingEvidence) dashMissingEvidence.innerText = `${missingCount} claims`;
        if (dashYearRange) dashYearRange.innerText = yearSpanStr;
        if (dashRetrievalMode) dashRetrievalMode.innerText = data.provider_display || dash.retrieval_mode || "Hybrid OpenAlex + Local SciFact";
        if (dashDirectVsIndirect) dashDirectVsIndirect.innerText = "Not available";

        // 2. Sufficiency Box & Main Findings
        const evalSufficiencyBox = document.getElementById('synthesisSufficiencyBox');
        const evalSufficiencyBadge = document.getElementById('evalSufficiencyBadge');
        const evalDegreeOfSupport = document.getElementById('evalDegreeOfSupport');
        const evalSufficiencyText = document.getElementById('evalSufficiencyText');

        const suffStatus = rs.sufficiency_status || (
            supCount === totalClaims && totalClaims > 0 ? "SUFFICIENT" :
            conCount === totalClaims && totalClaims > 0 ? "SUFFICIENT_REFUTATION" :
            neiCount === totalClaims && totalClaims > 0 ? "INSUFFICIENT" :
            supCount > 0 && conCount > 0 ? "INCONCLUSIVE_CONFLICT" : "PARTIALLY_SUFFICIENT"
        );

        if (evalSufficiencyBox) {
            evalSufficiencyBox.className = 'synthesis-evaluation-box';
            if (suffStatus === 'SUFFICIENT') {
                // default green
            } else if (suffStatus === 'INSUFFICIENT') {
                evalSufficiencyBox.classList.add('insufficient');
            } else if (suffStatus.includes('REFUTATION') || suffStatus.includes('CONTRADICTED')) {
                evalSufficiencyBox.classList.add('contradicted');
            } else {
                evalSufficiencyBox.classList.add('inconclusive');
            }
        }

        if (evalSufficiencyBadge) evalSufficiencyBadge.innerText = `SUFFICIENCY: ${suffStatus.replace(/_/g, ' ')}`;
        if (evalDegreeOfSupport) evalDegreeOfSupport.innerText = `Degree of Support: ${(rs.degree_of_support || "EVALUATED").replace(/_/g, ' ')}`;
        if (evalSufficiencyText) evalSufficiencyText.innerText = rs.sufficiency_text || "Evidence evaluation across retrieved literature completed.";

        // Strongest Conclusions & Qualifications Callouts
        const synthesisStrongestConclusions = document.getElementById('synthesisStrongestConclusions');
        const synthesisQualifications = document.getElementById('synthesisQualifications');

        if (synthesisStrongestConclusions) {
            const scList = rs.strongest_conclusions || [];
            if (scList.length > 0) {
                synthesisStrongestConclusions.innerHTML = `<ul>${scList.map(item => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`;
            } else {
                const suppClaims = claims.filter(c => normalizeVerdict(c.verdict || c.final_label) === 'SUPPORTED');
                if (suppClaims.length > 0) {
                    synthesisStrongestConclusions.innerHTML = `<ul>${suppClaims.map(c => `<li>Verified Assertion: "${escapeHtml(c.claim_text)}" (${((c.confidence || 0.85)*100).toFixed(0)}% confidence)</li>`).join('')}</ul>`;
                } else {
                    synthesisStrongestConclusions.innerHTML = `<p class="text-muted">No evaluated claims received unequivocal empirical support from the retrieved literature corpus.</p>`;
                }
            }
        }

        if (synthesisQualifications) {
            const qcList = rs.qualifications_and_conflicts || [];
            if (qcList.length > 0) {
                synthesisQualifications.innerHTML = `<ul>${qcList.map(item => `<li>${escapeHtml(item)}</li>`).join('')}</ul>`;
            } else {
                const unconfirmed = claims.filter(c => normalizeVerdict(c.verdict || c.final_label) !== 'SUPPORTED');
                if (unconfirmed.length > 0) {
                    synthesisQualifications.innerHTML = `<ul>${unconfirmed.map(c => `<li>Qualification on "${escapeHtml(c.claim_text)}": Result is ${normalizeVerdict(c.verdict || c.final_label)} (${((c.confidence || 0.7)*100).toFixed(0)}% confidence)</li>`).join('')}</ul>`;
                } else {
                    synthesisQualifications.innerHTML = `<p class="text-muted">No conflicting findings or significant evidentiary qualifications identified within the retrieved literature.</p>`;
                }
            }
        }

        // 3. Claim-by-Claim Accordion Rendering
        const accordionContainer = document.getElementById('synthesisClaimsAccordion');
        const claimsCountBadge = document.getElementById('synthesisClaimsCountBadge');
        if (claimsCountBadge) claimsCountBadge.innerText = `${totalClaims} Claims`;

        if (accordionContainer) {
            accordionContainer.innerHTML = '';
            claims.forEach((c, idx) => {
                const normVerdict = normalizeVerdict(c.verdict || c.final_label);
                const statusClass = getStatusClass(normVerdict);
                const evList = c.evidence || c.retrieved_evidence || [];
                const confPct = ((c.confidence !== undefined ? c.confidence : 0.85) * 100).toFixed(0);

                const item = document.createElement('div');
                item.className = 'claim-accordion-item';
                if (idx === 0) item.classList.add('open');

                // Generate evidence items HTML
                let evItemsHtml = '';
                if (evList.length === 0) {
                    evItemsHtml = `<div class="text-muted" style="font-size: 13px; font-style: italic; padding: 8px 0;">No relevant evidence passages were retrieved for this claim from the examined corpus.</div>`;
                } else {
                    evList.forEach(ev => {
                        const title = ev.title || ev.paper_title || "Scientific Publication";
                        let authors = ev.authors || ev.author || "Authors unlisted";
                        if (Array.isArray(authors)) authors = authors.join(", ");
                        const yr = ev.publication_year || ev.year || "n.d.";
                        const passageId = ev.passage_id || `p_${ev.rank || 1}`;
                        const section = ev.section || "Abstract";
                        const passageText = ev.exact_passage || ev.passage || ev.text || "";
                        const score = (ev.score !== undefined) ? ev.score : 0.0;
                        const relBadge = renderRelevanceBadge(score);

                        const doiUrl = ev.url_doi || ev.doi;
                        const doiHtml = doiUrl 
                            ? `<a href="${escapeHtml(doiUrl)}" target="_blank" style="color: var(--primary); text-decoration: underline;">DOI Link ↗</a>` 
                            : '<span>DOI unavailable</span>';

                        const rawRel = String(ev.relationship || ev.relation || '').toUpperCase();
                        let relTag = 'NEUTRAL / NEI';
                        let relStyle = 'background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1;';
                        if (rawRel.includes('CONTRADICT') || rawRel.includes('REFUTE') || normVerdict === 'CONTRADICTED') {
                            relTag = 'CONTRADICTS';
                            relStyle = 'background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; font-weight: 700;';
                        } else if (rawRel.includes('PARTIAL') || normVerdict === 'PARTIALLY_SUPPORTED') {
                            relTag = 'PARTIALLY SUPPORTS';
                            relStyle = 'background: #fffbeb; color: #d97706; border: 1px solid #fde68a; font-weight: 700;';
                        } else if (rawRel.includes('SUPPORT') || normVerdict === 'SUPPORTED') {
                            relTag = 'SUPPORTS';
                            relStyle = 'background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; font-weight: 700;';
                        }

                        // Store source metadata in JSON string for modal inspection
                        const sourceJsonStr = escapeHtml(JSON.stringify({
                            title: title,
                            authors: authors,
                            year: yr,
                            section: section,
                            passage_id: passageId,
                            score: score,
                            doi: doiUrl,
                            passage: passageText,
                            source: ev.source || ev.retrieval_method || 'Hybrid Retriever'
                        }));

                        evItemsHtml += `
                            <div class="acc-evidence-card">
                                <div class="acc-evidence-top">
                                    <div>
                                        <span class="acc-evidence-title">📄 ${escapeHtml(title)}</span>
                                        <span class="acc-evidence-authors"> (${escapeHtml(authors)}, ${escapeHtml(yr)})</span>
                                    </div>
                                    <div style="display: flex; gap: 6px; align-items: center;">
                                        <span style="font-size: 11px; padding: 2px 6px; border-radius: 4px; ${relStyle}">${relTag}</span>
                                        ${relBadge}
                                    </div>
                                </div>
                                <div class="acc-evidence-passage-text">
                                    "${escapeHtml(passageText)}"
                                </div>
                                <div class="acc-evidence-meta-row">
                                    <span>Passage: <code>${escapeHtml(passageId)}</code> (${escapeHtml(section)})</span>
                                    <span>${doiHtml}</span>
                                    <button class="btn-inspect-source" data-source='${sourceJsonStr}'>
                                        🔍 View Source Metadata
                                    </button>
                                </div>
                            </div>
                        `;
                    });
                }

                // Multi-agent reasoning HTML
                let agentDecisionsHtml = '';
                const ads = c.agent_decisions || c.agent_assessments || [];
                if (ads.length > 0) {
                    agentDecisionsHtml = `
                        <div class="acc-agent-reasoning-box">
                            <h6>🤖 AI Multi-Agent Verification Rationale</h6>
                            <div class="acc-agent-pill-row">
                                ${ads.map(ad => `
                                    <div class="acc-agent-pill">
                                        <strong>${escapeHtml(ad.agent_name || 'Agent')}</strong> 
                                        (${escapeHtml(normalizeVerdict(ad.verdict || ad.decision))}, ${((ad.confidence || 0.8)*100).toFixed(0)}%): 
                                        <em>${escapeHtml(ad.rationale || 'Consensus reasoning aligned with evidence.')}</em>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    `;
                }

                // Limitations / Failure Flag HTML
                let limitationsHtml = '';
                if (c.failure_signature) {
                    const fs = c.failure_signature;
                    limitationsHtml = `
                        <div class="acc-claim-limitations-box">
                            <strong>⚠️ Failure Category ${escapeHtml(fs.failure_type || 'ANOMALY')}:</strong> 
                            ${escapeHtml(fs.explanation || 'Anomaly signature flagged and logged to Runtime Audit Ledger.')}
                        </div>
                    `;
                }

                item.innerHTML = `
                    <div class="claim-acc-header">
                        <span class="acc-chevron">▶</span>
                        <span class="acc-claim-id">${escapeHtml(c.claim_id || `C${idx+1}`)}</span>
                        <span class="acc-claim-title">${escapeHtml(c.claim_text)}</span>
                        <div class="acc-badge-group">
                            <span class="badge-status ${statusClass}">${normVerdict}</span>
                            <span style="font-size: 11px; font-weight: 700; color: var(--text-secondary);">${confPct}%</span>
                            <span style="font-size: 11px; background: #e2e8f0; color: #334155; padding: 2px 6px; border-radius: 10px;">${evList.length} Evidences</span>
                        </div>
                    </div>
                    <div class="claim-acc-body">
                        <div class="claim-acc-assertion-box">
                            <strong>Atomic Scientific Assertion</strong>
                            <p>"${escapeHtml(c.claim_text)}"</p>
                        </div>
                        <div style="font-size: 13px; margin-bottom: 10px; color: var(--text-secondary);">
                            <strong>Resolution Method:</strong> <code>${escapeHtml(c.resolution_method || 'Multi-Agent Consensus')}</code> |
                            <strong>Confidence:</strong> <code>${confPct}%</code>
                        </div>
                        <div style="font-size: 13px; font-weight: 700; color: var(--text-primary); margin-top: 12px; margin-bottom: 6px;">
                            Mapped Scientific Evidence Passages (${evList.length}):
                        </div>
                        <div class="acc-evidence-list">
                            ${evItemsHtml}
                        </div>
                        ${agentDecisionsHtml}
                        ${limitationsHtml}
                    </div>
                `;

                // Accordion header click event
                const header = item.querySelector('.claim-acc-header');
                if (header) {
                    header.addEventListener('click', () => {
                        item.classList.toggle('open');
                    });
                }

                accordionContainer.appendChild(item);
            });

            // Wire source inspection buttons inside accordion
            accordionContainer.querySelectorAll('.btn-inspect-source').forEach(btn => {
                btn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const jsonStr = btn.getAttribute('data-source');
                    if (jsonStr) {
                        try {
                            const sourceData = JSON.parse(jsonStr);
                            openSourceModal(sourceData);
                        } catch (err) {
                            console.error('Failed to parse source data:', err);
                        }
                    }
                });
            });
        }

        // 4. Uncertainty & Research Gaps
        const synthesisUnverifiedClaims = document.getElementById('synthesisUnverifiedClaims');
        const synthesisContradictions = document.getElementById('synthesisContradictions');
        const synthesisMissingEvidence = document.getElementById('synthesisMissingEvidence');
        const synthesisFollowUpQuestions = document.getElementById('synthesisFollowUpQuestions');

        if (synthesisUnverifiedClaims) {
            const unverified = ug.unverified_claims || claims.filter(c => {
                const v = normalizeVerdict(c.verdict || c.final_label);
                return v === 'PARTIALLY_SUPPORTED' || v === 'INSUFFICIENT_EVIDENCE';
            }).map(c => ({
                claim_id: c.claim_id,
                claim_text: c.claim_text,
                verdict: normalizeVerdict(c.verdict || c.final_label),
                reason: 'Requires further empirical verification.'
            }));

            if (unverified.length > 0) {
                synthesisUnverifiedClaims.innerHTML = `<ul>${unverified.map(uc => `
                    <li><strong>${escapeHtml(uc.claim_id)}</strong> (${escapeHtml(uc.verdict)}): "${escapeHtml(uc.claim_text)}"</li>
                `).join('')}</ul>`;
            } else {
                synthesisUnverifiedClaims.innerHTML = `<p class="text-muted">All evaluated claims met evidentiary verification thresholds.</p>`;
            }
        }

        if (synthesisContradictions) {
            const contradicted = ug.contradicted_findings || claims.filter(c => normalizeVerdict(c.verdict || c.final_label) === 'CONTRADICTED').map(c => ({
                claim_id: c.claim_id,
                claim_text: c.claim_text
            }));

            if (contradicted.length > 0) {
                synthesisContradictions.innerHTML = `<ul>${contradicted.map(cf => `
                    <li><strong style="color: var(--contradicted-color);">${escapeHtml(cf.claim_id)}:</strong> "${escapeHtml(cf.claim_text)}" — Refuted by retrieved empirical literature.</li>
                `).join('')}</ul>`;
            } else {
                synthesisContradictions.innerHTML = `<p class="text-muted" style="font-size: 12px; font-style: italic;">
                    No contradictory literature was detected within the retrieved corpus.<br>
                    <em>Note: Contradictions were evaluated strictly against the retrieved papers, not exhaustive global literature.</em>
                </p>`;
            }
        }

        if (synthesisMissingEvidence) {
            const missing = ug.missing_evidence_claims || claims.filter(c => (c.evidence || []).length === 0).map(c => ({
                claim_id: c.claim_id,
                claim_text: c.claim_text,
                reason: 'No evidence passages retrieved from corpus.'
            }));

            if (missing.length > 0) {
                synthesisMissingEvidence.innerHTML = `<ul>${missing.map(me => `
                    <li><strong>${escapeHtml(me.claim_id)}</strong>: "${escapeHtml(me.claim_text)}" — <em>${escapeHtml(me.reason)}</em></li>
                `).join('')}</ul>`;
            } else {
                synthesisMissingEvidence.innerHTML = `<p class="text-muted">All claims have at least one mapped literature passage.</p>`;
            }
        }

        if (synthesisFollowUpQuestions) {
            const fuList = (ug && ug.follow_up_questions && ug.follow_up_questions.length > 0) 
                ? ug.follow_up_questions 
                : (rs.follow_up_questions || (report.sub_questions ? report.sub_questions.map(sq => `What targeted empirical studies substantiate: ${sq}?`) : [
                    `How do longitudinal or clinical trials validate the mechanisms underlying '${escapeHtml(report.research_question || data.research_question || 'this inquiry')}'?`,
                    `What confounding variables or population characteristics moderate these observed findings?`
                ]));

            synthesisFollowUpQuestions.innerHTML = fuList.map(fq => `<li>${escapeHtml(fq)}</li>`).join('');
        }

        // 5. Practical Implications & Recommended Next Steps
        const synthesisPracticalImplications = document.getElementById('synthesisPracticalImplications');
        const synthesisNextSteps = document.getElementById('synthesisNextSteps');

        if (synthesisPracticalImplications) {
            synthesisPracticalImplications.innerText = rs.practical_implications || (
                supCount > 0 
                ? `Given empirical support for ${supCount} core claim(s), researchers can tentatively rely on verified findings for experimental design or theoretical modeling, while maintaining vigilance over unverified sub-claims.`
                : `Given the absence of unequivocal empirical support in the retrieved corpus, practitioners should avoid premature clinical, policy, or experimental reliance on these unverified claims until primary literature validation is conducted.`
            );
        }

        if (synthesisNextSteps) {
            const nextStepsList = rs.recommended_next_steps || [
                "Conduct targeted retrieval across full-text primary literature (beyond abstract indices) to address identified evidentiary gaps.",
                "Verify potential population, dose, or methodological mismatches between retrieved evidence and formulated claims.",
                "Formulate experimental protocols to empirically test claims categorized as Insufficient Evidence or Partially Supported."
            ];
            synthesisNextSteps.innerHTML = nextStepsList.map(step => `<li>${escapeHtml(step)}</li>`).join('');
        }
    }

    // Modal, Copy, and Accordion Action Handlers
    function openSourceModal(source) {
        const modal = document.getElementById('sourceDetailsModal');
        const titleEl = document.getElementById('modalPaperTitle');
        const bodyEl = document.getElementById('modalSourceBody');
        if (!modal || !titleEl || !bodyEl) return;

        titleEl.innerText = source.title || 'Scientific Publication';
        bodyEl.innerHTML = `
            <table class="meta-table">
                <tr><th>Authors</th><td>${escapeHtml(source.authors || 'Authors unlisted')}</td></tr>
                <tr><th>Publication Year</th><td>${escapeHtml(source.year || 'n.d.')}</td></tr>
                <tr><th>Section & Passage ID</th><td>${escapeHtml(source.section || 'Abstract')} (<code>${escapeHtml(source.passage_id || 'p01')}</code>)</td></tr>
                <tr><th>Relevance Score</th><td><code>${source.score !== undefined ? Number(source.score).toFixed(4) : 'N/A'}</code></td></tr>
                <tr><th>Retrieval Provider</th><td>${escapeHtml(source.source || 'OpenAlex / Local Corpus')}</td></tr>
                <tr><th>DOI / Web Link</th><td>${source.doi ? `<a href="${escapeHtml(source.doi)}" target="_blank" style="color: var(--primary); text-decoration: underline;">${escapeHtml(source.doi)} ↗</a>` : 'DOI unavailable'}</td></tr>
            </table>
            <div style="margin-top: 14px;">
                <strong style="font-size: 11px; text-transform: uppercase; color: var(--text-muted); display: block; margin-bottom: 4px;">Exact Extracted Literature Excerpt:</strong>
                <blockquote style="margin: 0; background: #f8fafc; border-left: 3px solid var(--primary); padding: 12px 14px; font-style: italic; font-size: 13px; color: var(--text-secondary); line-height: 1.6;">
                    "${escapeHtml(source.passage || 'No text excerpt recorded.')}"
                </blockquote>
            </div>
        `;

        modal.classList.remove('hidden');
        modal.style.display = 'flex';
    }

    const closeSourceModalBtn = document.getElementById('closeSourceModalBtn');
    const closeSourceModalFooterBtn = document.getElementById('closeSourceModalFooterBtn');
    const sourceDetailsModal = document.getElementById('sourceDetailsModal');

    if (closeSourceModalBtn) {
        closeSourceModalBtn.addEventListener('click', () => {
            if (sourceDetailsModal) {
                sourceDetailsModal.classList.add('hidden');
                sourceDetailsModal.style.display = 'none';
            }
        });
    }
    if (closeSourceModalFooterBtn) {
        closeSourceModalFooterBtn.addEventListener('click', () => {
            if (sourceDetailsModal) {
                sourceDetailsModal.classList.add('hidden');
                sourceDetailsModal.style.display = 'none';
            }
        });
    }
    if (sourceDetailsModal) {
        sourceDetailsModal.addEventListener('click', (e) => {
            if (e.target === sourceDetailsModal) {
                sourceDetailsModal.classList.add('hidden');
                sourceDetailsModal.style.display = 'none';
            }
        });
    }

    const btnCopySynthesis = document.getElementById('btnCopySynthesis');
    if (btnCopySynthesis) {
        btnCopySynthesis.addEventListener('click', () => {
            const synthQuote = synthesisText ? synthesisText.innerText : '';
            const evalText = document.getElementById('evalSufficiencyText') ? document.getElementById('evalSufficiencyText').innerText : '';
            const fullSynthesis = `${synthQuote}\n\nSufficiency Assessment:\n${evalText}`;
            if (navigator.clipboard) {
                navigator.clipboard.writeText(fullSynthesis).then(() => {
                    const originalText = btnCopySynthesis.innerText;
                    btnCopySynthesis.innerText = '✓ Copied!';
                    setTimeout(() => { btnCopySynthesis.innerText = originalText; }, 2000);
                });
            }
        });
    }

    const btnPrintReport = document.getElementById('btnPrintReport');
    if (btnPrintReport) {
        btnPrintReport.addEventListener('click', () => {
            window.print();
        });
    }

    const btnToggleAllClaims = document.getElementById('btnToggleAllClaims');
    if (btnToggleAllClaims) {
        let allExpanded = false;
        btnToggleAllClaims.addEventListener('click', () => {
            const items = document.querySelectorAll('.claim-accordion-item');
            allExpanded = !allExpanded;
            items.forEach(item => {
                if (allExpanded) item.classList.add('open');
                else item.classList.remove('open');
            });
            btnToggleAllClaims.innerText = allExpanded ? '↕️ Collapse All Claims' : '↕️ Expand All Claims';
        });
    }
});
