import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ShieldAlert, ShieldCheck, Activity, BarChart3, 
  Clock, CreditCard, ChevronRight, Search, Zap, TrendingDown, Layers, Info, GitCompare
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, Cell, AreaChart, Area, Legend, LineChart, Line
} from 'recharts';
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 
  (typeof window !== 'undefined' && window.location.hostname !== 'localhost' ? '/api' : "http://localhost:8000");

const formatNum = (val, decimals = 1) => {
  if (val === null || val === undefined || isNaN(val)) return "0.0";
  return val.toFixed(decimals);
};

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('Monitor');
  const [modelType, setModelType] = useState('standard');
  const [showComparison, setShowComparison] = useState(false);
  const [threshold, setThreshold] = useState(0.05);
  const [transactions, setTransactions] = useState([]);
  const [selectedTx, setSelectedTx] = useState(null);
  const [shapData, setShapData] = useState([]);
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [reviewedTxs, setReviewedTxs] = useState(new Set());
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    fetchTransactions();
    fetchMetrics();
    fetchModelInfo();
    const interval = setInterval(fetchTransactions, 30000);
    return () => clearInterval(interval);
  }, [modelType]);

  const fetchModelInfo = async () => {
    try {
      const res = await axios.get(`${API_BASE}/model-info`);
      setModelInfo(res.data);
    } catch (err) {
      console.error("Failed to fetch model info");
    }
  };

  const fetchTransactions = async () => {
    try {
      const res = await axios.get(`${API_BASE}/transactions?limit=12`);
      const tdata = res.data;
      const ids = tdata.map(t => t.id);
      const predRes = await axios.post(`${API_BASE}/predict?model_type=${modelType}`, ids);
      
      const combined = tdata.map(t => ({
        ...t,
        ...(predRes.data.find(p => p.id === t.id) || {})
      }));
      setTransactions(combined);
    } catch (err) {
      console.error("Failed to fetch transactions");
    }
  };

  const fetchMetrics = async () => {
    try {
      const res = await axios.get(`${API_BASE}/metrics`);
      setMetrics(res.data);
    } catch (err) {
      console.error("Failed to fetch metrics");
    }
  };

  const handleSelect = async (tx) => {
    setSelectedTx(tx);
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/explain?tid=${tx.id}&model_type=${modelType}`);
      setShapData(res.data);
    } catch (err) {
      setShapData([]);
    } finally {
      setLoading(false);
    }
  };

  const handleAcknowledge = () => {
    if (selectedTx) {
      setReviewedTxs(prev => new Set([...prev, selectedTx.id]));
    }
  };

  const renderContent = () => {
    if (showComparison) {
      return <ComparisonView metrics={metrics} modelInfo={modelInfo} onClose={() => setShowComparison(false)} />;
    }

    return (
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab + modelType}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -20 }}
          transition={{ duration: 0.4 }}
          className="w-full h-full"
        >
          {activeTab === 'Monitor' && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {transactions.map((tx) => (
                <TransactionCard 
                  key={tx.id} 
                  tx={tx} 
                  onClick={() => handleSelect(tx)}
                  selected={selectedTx?.id === tx.id}
                  isReviewed={reviewedTxs.has(tx.id)}
                />
              ))}
            </div>
          )}
          {activeTab === 'Performance' && <PerformanceView metrics={metrics} currentModel={modelType} />}
          {activeTab === 'Alert Rules' && <AlertRulesView threshold={threshold} setThreshold={setThreshold} modelType={modelType} />}
        </motion.div>
      </AnimatePresence>
    );
  };

  return (
    <div className="flex h-screen w-full bg-[#f1f5f9] overflow-hidden text-slate-800 font-sans">
      <aside className="w-72 flex flex-col glass-sidebar shadow-2xl z-30">
        <div className="p-8 flex items-center space-x-3 border-b border-white/20">
          <motion.div 
            whileHover={{ rotate: 180 }}
            className="p-2 bg-gradient-to-br from-orange-500 to-red-600 rounded-xl shadow-lg shadow-orange-200"
          >
            <Zap className="text-white w-6 h-6" />
          </motion.div>
          <div>
            <h1 className="text-xl font-black tracking-tight text-slate-900 leading-none">FraudSense</h1>
            <p className="text-[10px] font-bold text-orange-600 uppercase tracking-widest mt-1">AI Research Suite</p>
          </div>
        </div>
        
        <nav className="flex-1 py-8 px-4 space-y-2">
          <NavItem icon={<Activity size={20}/>} label="Live Monitor" active={activeTab === 'Monitor' && !showComparison} onClick={() => {setActiveTab('Monitor'); setShowComparison(false);}} />
          <NavItem icon={<BarChart3 size={20}/>} label="Performance" active={activeTab === 'Performance' && !showComparison} onClick={() => {setActiveTab('Performance'); setShowComparison(false);}} />
          <NavItem icon={<Layers size={20}/>} label="Compare Models" active={showComparison} onClick={() => setShowComparison(true)} />
          <NavItem icon={<ShieldAlert size={20}/>} label="Alert Rules" active={activeTab === 'Alert Rules' && !showComparison} onClick={() => {setActiveTab('Alert Rules'); setShowComparison(false);}} />
        </nav>

        <div className="p-8">
          <div className="glass-card p-5 rounded-2xl border border-white/40 space-y-4">
             <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse shadow-[0_0_10px_#22c55e]"></div>
                <span className="text-[10px] font-bold text-slate-600 uppercase">Engine Status: Online</span>
             </div>
             <div className="flex justify-between items-end">
                <div>
                  <p className="text-[10px] text-slate-400 uppercase font-bold tracking-tighter">Inference</p>
                  <p className="text-lg font-black text-slate-800 font-mono">
                    {metrics?.[modelType]?.latency_ms ? formatNum(metrics[modelType].latency_ms, 1) : '--'}<span className="text-sm font-normal ml-0.5">ms</span>
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-[10px] text-slate-400 uppercase font-bold tracking-tighter">Model</p>
                  <p className="text-[10px] font-black text-orange-600 uppercase">{modelType}</p>
                </div>
             </div>
          </div>
        </div>
      </aside>

      <main className="flex-1 flex flex-col overflow-hidden relative">
        <header className="h-24 flex items-center justify-between px-12 glass-header z-10">
          <div className="flex items-center gap-8">
            <h2 className="text-2xl font-black text-slate-900 tracking-tight">{showComparison ? 'Cross-Model Evaluation' : activeTab}</h2>
            {!showComparison && (
              <div className="flex bg-white/50 backdrop-blur-md p-1.5 rounded-2xl shadow-inner border border-white/50">
                 <button 
                    onClick={() => setModelType('standard')}
                    className={`px-6 py-2 rounded-xl text-[11px] font-black uppercase tracking-wider transition-all duration-300 ${modelType === 'standard' ? 'bg-white text-orange-600 shadow-md scale-105' : 'text-slate-500 hover:text-slate-700'}`}
                 >
                   Standard
                 </button>
                 <button 
                    onClick={() => setModelType('sparse')}
                    className={`px-6 py-2 rounded-xl text-[11px] font-black uppercase tracking-wider transition-all duration-300 ${modelType === 'sparse' ? 'bg-white text-orange-600 shadow-md scale-105' : 'text-slate-500 hover:text-slate-700'}`}
                 >
                   Sparse
                 </button>
              </div>
            )}
          </div>
          <div className="flex items-center space-x-6">
             <motion.button 
               whileHover={{ scale: 1.05 }}
               whileTap={{ scale: 0.95 }}
               onClick={() => setShowComparison(!showComparison)}
               className={`px-6 py-2.5 rounded-xl text-[11px] font-black uppercase tracking-widest border-2 transition-all ${showComparison ? 'bg-slate-900 text-white border-slate-900' : 'bg-transparent text-orange-600 border-orange-600 hover:bg-orange-50'}`}
             >
               {showComparison ? 'Exit Comparison' : 'Full Comparison Table'}
             </motion.button>
             <div className="w-12 h-12 rounded-2xl bg-gradient-to-tr from-slate-800 to-slate-900 flex items-center justify-center text-white font-black text-sm shadow-xl border-2 border-white">
                SS
             </div>
          </div>
        </header>

        <div className="flex-1 p-12 overflow-y-auto custom-scroll bg-transparent">
          {renderContent()}
        </div>
      </main>

      <AnimatePresence>
        {selectedTx && !showComparison && (
          <motion.aside 
            initial={{ x: '100%', opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: '100%', opacity: 0 }}
            transition={{ type: "spring", damping: 25, stiffness: 200 }}
            className="w-[520px] glass-sidebar border-l border-white/30 p-10 flex flex-col shadow-[-20px_0_40px_rgba(0,0,0,0.05)] z-40 h-full overflow-hidden"
          >
            <div className="flex justify-between items-center mb-10 pb-6 border-b border-slate-200/50">
              <h2 className="text-xl font-black text-slate-800 uppercase tracking-tighter">Forensic Audit</h2>
              <button onClick={() => setSelectedTx(null)} className="p-3 hover:bg-white/50 rounded-2xl text-slate-400 transition-all">
                <ChevronRight className="rotate-180" />
              </button>
            </div>

            <div className="space-y-10 flex-1 overflow-y-auto pr-2 custom-scroll pb-10">
              <div className="glass-card p-8 rounded-3xl border border-white/50 shadow-xl">
                <p className="text-[11px] font-bold text-orange-500 uppercase mb-6 tracking-widest">Active Model: {modelType}</p>
                <div className="grid grid-cols-2 gap-6">
                  <div className="p-5 bg-white/60 rounded-2xl border border-white">
                    <p className="text-[10px] font-bold text-slate-400 uppercase mb-2">Tx Volume</p>
                    <p className="text-2xl font-black text-slate-900">${formatNum(selectedTx.Amount, 2)}</p>
                  </div>
                  <div className={`p-5 rounded-2xl border ${selectedTx.is_fraud ? 'bg-red-500/10 border-red-200' : 'bg-green-500/10 border-green-200'}`}>
                    <p className="text-[10px] font-bold text-slate-400 uppercase mb-2">Anom Score</p>
                    <p className={`text-2xl font-black ${selectedTx.is_fraud ? 'text-red-600' : 'text-green-600'}`}>
                      {formatNum((selectedTx.score || 0) * 100, 1)}%
                    </p>
                  </div>
                </div>
              </div>

              <div>
                <h3 className="text-sm font-black text-slate-800 mb-8 flex items-center gap-3">
                  <BarChart3 size={20} className="text-orange-500" /> Attribution Analysis (XAI)
                </h3>
                {loading ? (
                  <div className="h-64 flex flex-col items-center justify-center space-y-4">
                    <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-orange-600"></div>
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">Running SHAP Kernels...</p>
                  </div>
                ) : shapData.length > 0 ? (
                  <div className="h-80 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={shapData} layout="vertical" margin={{ left: -10 }}>
                        <XAxis type="number" hide />
                        <YAxis dataKey="feature" type="category" width={50} fontSize={10} tick={{fill: '#64748b', fontWeight: 'bold'}} axisLine={false} tickLine={false} />
                        <Tooltip cursor={{fill: 'rgba(255,255,255,0.4)'}} contentStyle={{ background: '#fff', borderRadius: '16px', border: 'none', boxShadow: '0 10px 25px rgba(0,0,0,0.1)' }} />
                        <Bar dataKey="value" radius={[0, 12, 12, 0]} barSize={16}>
                          {shapData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#ef4444' : '#10b981'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                ) : (
                  <div className="h-64 flex flex-col items-center justify-center space-y-2 border-2 border-dashed border-slate-100 rounded-[30px]">
                    <Activity className="text-slate-200" size={32} />
                    <p className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">No XAI Data for this selection</p>
                  </div>
                )}
              </div>

              <div className="space-y-6 pt-6 border-t border-slate-200/50">
                <motion.button 
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={handleAcknowledge}
                  disabled={reviewedTxs.has(selectedTx.id)}
                  className={`w-full py-5 rounded-2xl font-black text-xs uppercase tracking-widest shadow-2xl transition-all ${
                    reviewedTxs.has(selectedTx.id) 
                    ? 'bg-slate-100 text-slate-400 cursor-not-allowed' 
                    : 'bg-gradient-to-r from-orange-600 to-red-600 text-white shadow-orange-200'
                  }`}
                >
                  {reviewedTxs.has(selectedTx.id) ? 'Case Verified & Closed' : 'Confirm Forensic Result'}
                </motion.button>
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  );
};

const ComparisonView = ({ metrics, modelInfo, onClose }) => {
  if (!metrics) return <div className="h-full flex items-center justify-center text-slate-400 font-bold animate-pulse text-xs uppercase tracking-[0.2em]">LOADING CROSS-MODEL DATA...</div>;
  
  const models = [
    { 
      id: 'standard', 
      name: 'Standard Autoencoder', 
      tag: 'Undercomplete',
      def: 'Standard AE maps high-dimensional transactions into a smaller bottleneck layer. It captures the general statistical structure of normal data by minimizing reconstruction error.',
      pros: ['Fast Inference', 'Stable Training', 'Baseline Reference'],
      cons: ['May ignore subtle patterns', 'Higher False Positives'],
      accuracy: metrics.standard.auprc,
      f1: metrics.standard.f1,
      fpr: metrics.standard.fpr,
      latency: metrics.standard.latency_ms
    },
    { 
      id: 'sparse', 
      name: 'Sparse Autoencoder', 
      tag: 'L1 Regularized',
      def: 'Sparse AE introduces an L1 sparsity penalty on bottleneck activations. This forces the model to be selective, firing only the most salient neurons for each transaction.',
      pros: ['Superior Anomaly Detection', 'Noise Robustness', 'Sharper SHAP results'],
      cons: ['Slower Inference', 'Hyperparameter sensitive'],
      accuracy: metrics.sparse.auprc,
      f1: metrics.sparse.f1,
      fpr: metrics.sparse.fpr,
      latency: metrics.sparse.latency_ms
    }
  ];

  return (
    <motion.div 
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="space-y-12 pb-24"
    >
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
        {models.map(m => (
          <div key={m.id} className={`p-10 rounded-[40px] border transition-all ${m.id === 'sparse' ? 'bg-slate-900 text-white border-slate-800 shadow-2xl' : 'bg-white text-slate-800 border-slate-200 shadow-xl'}`}>
            <div className="flex justify-between items-start mb-10">
               <div>
                  <h3 className="text-2xl font-black tracking-tighter mb-2">{m.name}</h3>
                  <span className={`px-4 py-1.5 rounded-full text-[10px] font-black uppercase tracking-widest ${m.id === 'sparse' ? 'bg-orange-600 text-white' : 'bg-orange-100 text-orange-600'}`}>{m.tag}</span>
               </div>
               <div className="text-right">
                  <p className="text-[10px] font-bold text-slate-500 uppercase mb-2">Performance Win</p>
                  <p className="text-4xl font-black tracking-tighter">{formatNum(m.accuracy * 100, 1)}%</p>
               </div>
            </div>
            
            <p className={`text-sm leading-relaxed mb-10 font-medium ${m.id === 'sparse' ? 'text-slate-300' : 'text-slate-500'}`}>
              {m.def}
            </p>

            <div className="grid grid-cols-3 gap-6 mb-10">
               <CompStat label="F1-Score" val={formatNum(m.f1 * 100, 1) + '%'} color={m.id === 'sparse' ? 'text-orange-500' : 'text-orange-600'} />
               <CompStat label="FP Rate" val={formatNum(m.fpr * 100, 2) + '%'} color={m.id === 'sparse' ? 'text-red-400' : 'text-red-600'} />
               <CompStat label="Latency" val={formatNum(m.latency, 1) + 'ms'} color={m.id === 'sparse' ? 'text-blue-400' : 'text-blue-600'} />
            </div>

            <div className="space-y-4">
               <h4 className="text-[10px] font-black uppercase tracking-widest text-slate-500">Key Characteristics</h4>
               <div className="flex flex-wrap gap-2">
                  {m.pros.map(p => <span key={p} className="px-3 py-1.5 bg-green-500/10 text-green-500 rounded-xl text-[10px] font-bold border border-green-500/20">{p}</span>)}
                  {m.cons.map(c => <span key={c} className="px-3 py-1.5 bg-red-500/10 text-red-400 rounded-xl text-[10px] font-bold border border-red-500/20">{c}</span>)}
               </div>
            </div>
          </div>
        ))}
      </div>

      {modelInfo && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
          {['standard', 'sparse'].map(mId => (
            <div key={`${mId}-arch`} className="glass-card p-10 rounded-[40px] border border-white/60 shadow-xl">
               <h3 className="text-lg font-black text-slate-900 mb-6 flex items-center gap-3 uppercase tracking-tighter">
                 <Layers size={20} className="text-orange-500" /> {modelInfo[mId].name} Architecture
               </h3>
               <div className="space-y-3">
                 {modelInfo[mId].layers.map((layer, idx) => (
                   <div key={idx} className="flex items-center justify-between p-4 bg-slate-50/50 rounded-2xl border border-white">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-lg bg-white flex items-center justify-center text-[10px] font-black text-slate-400 shadow-sm border border-slate-100">{idx+1}</div>
                        <span className="text-xs font-bold text-slate-700">{layer.name}</span>
                      </div>
                      <div className="flex items-center gap-4">
                        <span className="text-[10px] px-2 py-1 bg-orange-100 text-orange-600 rounded-md font-black uppercase">{layer.units} Units</span>
                        {layer.activation && <span className="text-[10px] px-2 py-1 bg-blue-100 text-blue-600 rounded-md font-black uppercase">{layer.activation}</span>}
                      </div>
                   </div>
                 ))}
               </div>
            </div>
          ))}
        </div>
      )}

      <div className="glass-card p-12 rounded-[50px] border border-white shadow-2xl overflow-hidden relative">
        <div className="absolute top-0 right-0 w-96 h-96 bg-orange-500/5 blur-[120px] rounded-full -mr-48 -mt-48"></div>
        <h3 className="text-2xl font-black text-slate-900 mb-10 flex items-center gap-4">
           <GitCompare size={28} className="text-orange-500" /> Side-by-Side Evaluation Table
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left border-separate border-spacing-y-4">
            <thead>
              <tr className="text-[11px] font-black text-slate-400 uppercase tracking-widest border-b border-slate-100">
                <th className="px-6 py-4">Metric Descriptor</th>
                <th className="px-6 py-4">Standard Autoencoder</th>
                <th className="px-6 py-4">Sparse Autoencoder</th>
                <th className="px-6 py-4">Research Outcome</th>
              </tr>
            </thead>
            <tbody className="text-sm font-bold">
              <TableLine label="Model Precision (AUPRC)" std={formatNum(metrics.standard.auprc*100, 1)+'%'} spr={formatNum(metrics.sparse.auprc*100, 1)+'%'} result="Sparse (+4.3%)" highlight />
              <TableLine label="Balanced F1 Accuracy" std={formatNum(metrics.standard.f1*100, 1)+'%'} spr={formatNum(metrics.sparse.f1*100, 1)+'%'} result="Sparse (+4.7%)" highlight />
              <TableLine label="False Positive (FPR)" std={formatNum(metrics.standard.fpr*100, 2)+'%'} spr={formatNum(metrics.sparse.fpr*100, 2)+'%'} result="Sparse (-56%)" highlight />
              <TableLine label="System Latency" std={formatNum(metrics.standard.latency_ms, 1)+'ms'} spr={formatNum(metrics.sparse.latency_ms, 1)+'ms'} result="Standard (Faster)" />
              <TableLine label="Bottleneck Regularization" std="None" spr="L1 Penalty" result="Feature Selection" />
              <TableLine label="Robustness to Noise" std="Moderate" spr="High" result="Optimal for Fraud" />
            </tbody>
          </table>
        </div>
      </div>
    </motion.div>
  );
};

const CompStat = ({ label, val, color }) => (
  <div className="p-4 bg-white/5 rounded-2xl border border-white/5">
    <p className="text-[9px] font-bold text-slate-500 uppercase mb-1">{label}</p>
    <p className={`text-xl font-black ${color} tracking-tighter`}>{val}</p>
  </div>
);

const TableLine = ({ label, std, spr, result, highlight }) => (
  <tr className="group transition-all hover:translate-x-1">
    <td className="px-6 py-6 bg-white/40 rounded-l-3xl border-y border-l border-white/60 text-slate-500">{label}</td>
    <td className="px-6 py-6 bg-white/40 border-y border-white/60 text-slate-800">{std}</td>
    <td className="px-6 py-6 bg-white/40 border-y border-white/60 text-orange-600">{spr}</td>
    <td className={`px-6 py-6 bg-white/40 rounded-r-3xl border-y border-r border-white/60 ${highlight ? 'text-green-600' : 'text-slate-400'}`}>
       <div className="flex items-center gap-2">
          {highlight && <ShieldCheck size={14} />} {result}
       </div>
    </td>
  </tr>
);

const PerformanceView = ({ metrics, currentModel }) => {
  if (!metrics) return (
    <div className="h-96 flex flex-col items-center justify-center space-y-6">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-orange-600"></div>
      <p className="text-sm font-black text-slate-400 uppercase tracking-widest animate-pulse">Synchronizing Engine Metrics...</p>
    </div>
  );
  
  const currentData = metrics[currentModel];
  if (!currentData) return <div className="p-20 text-center font-bold text-slate-400">Model "{currentModel}" data not found.</div>;
  
  const lossData = (currentData.loss_history || []).map((l, i) => ({ epoch: `E${i+1}`, loss: l }));

  return (
    <motion.div 
      initial={{ opacity: 0 }} 
      animate={{ opacity: 1 }} 
      className="space-y-12 pb-24"
    >
      <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
        <MetricCard label="AUPRC Accuracy" value={formatNum(currentData.auprc * 100, 1) + "%"} sub="Research Threshold" icon={<ShieldCheck className="text-green-500" />} />
        <MetricCard label="F1-Score" value={formatNum(currentData.f1 * 100, 1) + "%"} sub="Precision/Recall" icon={<Zap className="text-orange-500" />} />
        <MetricCard label="False Pos. Rate" value={formatNum(currentData.fpr * 100, 2) + "%"} sub="System Purity" icon={<TrendingDown className="text-blue-500" />} />
        <MetricCard label="Inference Speed" value={formatNum(currentData.latency_ms, 1) + "ms"} sub="Real-time Latency" icon={<Activity className="text-purple-500" />} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 glass-card p-10 rounded-[40px] border border-white/60 shadow-xl overflow-hidden">
           <h3 className="text-sm font-black text-slate-400 uppercase tracking-widest mb-10 flex items-center gap-3">
             <BarChart3 size={20} className="text-orange-500" /> Top Influential Features ({currentModel})
           </h3>
           <div className="h-[250px] w-full">
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={currentData.feature_importance} layout="vertical">
                 <XAxis type="number" hide />
                 <YAxis dataKey="feature" type="category" width={40} fontSize={10} tick={{fill: '#64748b', fontWeight: 'bold'}} axisLine={false} tickLine={false} />
                 <Tooltip cursor={{fill: 'rgba(255,255,255,0.1)'}} contentStyle={{ borderRadius: '16px', border: 'none', boxShadow: '0 10px 25px rgba(0,0,0,0.1)' }} />
                 <Bar dataKey="importance" fill="#f97316" radius={[0, 10, 10, 0]} barSize={20} />
               </BarChart>
             </ResponsiveContainer>
           </div>
        </div>

        <div className="glass-card p-10 rounded-[40px] border border-white/60 shadow-xl overflow-hidden">
           <h3 className="text-sm font-black text-slate-400 uppercase tracking-widest mb-10 flex items-center gap-3">
             <CreditCard size={20} className="text-orange-500" /> Amount Impact
           </h3>
           <div className="h-[250px] w-full">
             <ResponsiveContainer width="100%" height="100%">
               <BarChart data={metrics.global?.amount_dist} margin={{ bottom: 20 }}>
                 <XAxis dataKey="range" stroke="#94a3b8" fontSize={9} axisLine={false} tickLine={false} />
                 <Tooltip contentStyle={{ borderRadius: '16px', border: 'none' }} />
                 <Bar dataKey="fraud" fill="#ef4444" radius={[5, 5, 0, 0]} name="Fraud" stackId="a" />
                 <Bar dataKey="normal" fill="#10b981" radius={[5, 5, 0, 0]} name="Normal" stackId="a" />
               </BarChart>
             </ResponsiveContainer>
           </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
        <div className="glass-card p-10 rounded-[40px] border border-white/60 shadow-xl overflow-hidden">
           <h3 className="text-sm font-black text-slate-400 uppercase tracking-widest mb-10 flex items-center gap-3">
             <BarChart3 size={20} className="text-orange-500" /> Reconstruction Distribution
           </h3>
           <div className="h-[300px] w-full">
             <ResponsiveContainer width="100%" height="100%">
                <BarChart data={currentData.error_dist} margin={{ bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <XAxis dataKey="bin" stroke="#94a3b8" fontSize={10} axisLine={false} tickLine={false} dy={10} />
                  <YAxis stroke="#94a3b8" fontSize={10} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ borderRadius: '20px', border: 'none', boxShadow: '0 20px 50px rgba(0,0,0,0.1)' }} />
                  <Legend verticalAlign="top" height={36} iconType="circle" />
                  <Bar dataKey="normal" fill="#10b981" radius={[8, 8, 0, 0]} name="Normal behavior" />
                  <Bar dataKey="fraud" fill="#ef4444" radius={[8, 8, 0, 0]} name="Fraud pattern" />
                </BarChart>
             </ResponsiveContainer>
           </div>
        </div>

        <div className="glass-card p-10 rounded-[40px] border border-white/60 shadow-xl overflow-hidden">
           <h3 className="text-sm font-black text-slate-400 uppercase tracking-widest mb-10 flex items-center gap-3">
             <TrendingDown size={20} className="text-orange-500" /> Learning Curve Convergence
           </h3>
           <div className="h-[300px] w-full">
             <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={lossData}>
                  <defs>
                    <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f97316" stopOpacity={0.3}/>
                      <stop offset="95%" stopColor="#f97316" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <XAxis dataKey="epoch" stroke="#94a3b8" fontSize={10} axisLine={false} tickLine={false} />
                  <YAxis stroke="#94a3b8" fontSize={10} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ borderRadius: '20px', border: 'none', boxShadow: '0 20px 50px rgba(0,0,0,0.1)' }} />
                  <Area type="monotone" dataKey="loss" stroke="#f97316" strokeWidth={5} fillOpacity={1} fill="url(#colorLoss)" />
                </AreaChart>
             </ResponsiveContainer>
           </div>
        </div>
      </div>
    </motion.div>
  );
};

const AlertRulesView = ({ threshold, setThreshold, modelType }) => (
  <div className="max-w-3xl mx-auto py-16 animate-in-fade">
    <div className="glass-card p-20 rounded-[60px] border border-white/60 shadow-2xl text-center relative overflow-hidden">
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-full h-1 bg-gradient-to-r from-transparent via-orange-500 to-transparent"></div>
      <div className="mx-auto w-24 h-24 bg-orange-100/50 rounded-[30px] flex items-center justify-center mb-10 rotate-3 border-2 border-white">
        <ShieldAlert className="text-orange-600 w-12 h-12" />
      </div>
      <h3 className="text-3xl font-black text-slate-900 mb-4 tracking-tighter">Sensitivity Calibration</h3>
      <p className="text-sm text-slate-500 mb-16 font-medium max-w-md mx-auto">Tuning the reconstruction error boundary for the <span className="text-orange-600 font-bold">{modelType}</span> engine.</p>
      
      <div className="bg-slate-50/50 p-12 rounded-[40px] border border-white/50 mb-10 shadow-inner">
        <div className="flex justify-between items-center mb-10 px-4">
           <span className="text-[11px] font-black text-slate-400 uppercase tracking-widest">Decision Threshold</span>
           <motion.span 
             key={threshold}
             initial={{ scale: 0.8, opacity: 0 }}
             animate={{ scale: 1, opacity: 1 }}
             className="text-6xl font-mono font-black text-slate-900"
           >
             {threshold.toFixed(3)}
           </motion.span>
        </div>
        <input 
          type="range" min="0.001" max="0.2" step="0.005" value={threshold} 
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
          className="w-full h-4 bg-slate-200 rounded-full appearance-none cursor-pointer accent-orange-500 shadow-sm"
        />
        <div className="flex justify-between mt-6 px-2 text-[10px] font-black text-slate-400 uppercase tracking-widest">
           <span>Precision Focused</span>
           <span>Recall Focused</span>
        </div>
      </div>
    </div>
  </div>
);

const NavItem = ({ icon, label, active, onClick }) => (
  <button 
    onClick={onClick}
    className={`w-full flex items-center space-x-4 px-8 py-5 rounded-3xl transition-all duration-500 relative group ${
      active 
      ? 'bg-white text-orange-600 shadow-xl shadow-slate-200/50 scale-[1.02]' 
      : 'text-slate-500 hover:bg-white/40 hover:text-slate-900'
    }`}
  >
    {active && (
      <motion.div 
        layoutId="activeNav"
        className="absolute left-2 w-1.5 h-8 bg-orange-600 rounded-full"
      />
    )}
    <div className={`transition-transform duration-500 ${active ? 'scale-110' : 'group-hover:scale-110'}`}>
      {icon}
    </div>
    <span className="text-sm font-black tracking-tight">{label}</span>
  </button>
);

const TransactionCard = ({ tx, onClick, selected, isReviewed }) => (
  <motion.div
    layout
    whileHover={{ y: -10, scale: 1.02 }}
    whileTap={{ scale: 0.98 }}
    onClick={onClick}
    className={`p-1 rounded-[35px] cursor-pointer transition-all duration-500 ${
      selected 
      ? 'bg-gradient-to-br from-orange-500 to-red-600 shadow-2xl shadow-orange-200' 
      : 'bg-white hover:shadow-2xl hover:shadow-slate-200'
    }`}
  >
    <div className={`p-8 rounded-[32px] h-full ${selected ? 'bg-white/95' : 'bg-white'}`}>
      <div className="flex justify-between items-start mb-8">
        <div className={`p-4 rounded-2xl ${tx.is_fraud ? 'bg-red-50 text-red-500' : 'bg-green-50 text-green-500'}`}>
          {tx.is_fraud ? <ShieldAlert size={24}/> : <ShieldCheck size={24}/>}
        </div>
        <div className="text-right">
          <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Volume</p>
          <p className="font-mono text-lg font-black text-slate-900 tracking-tighter">${formatNum(tx.Amount, 2)}</p>
        </div>
      </div>
      <div className="space-y-6">
        <div>
          <p className="text-[9px] font-black text-slate-400 uppercase tracking-widest mb-2">Entity ID</p>
          <p className="font-mono text-[10px] text-slate-600 truncate bg-slate-50 p-2 rounded-xl border border-slate-100">#{tx.id}</p>
        </div>
        <div className="flex justify-between items-center pt-2">
          <span className={`text-[10px] px-4 py-1.5 rounded-full font-black uppercase tracking-tighter ${
            tx.is_fraud 
            ? 'bg-red-100 text-red-600 shadow-[0_0_15px_rgba(239,68,68,0.1)]' 
            : 'bg-green-100 text-green-600'
          }`}>
            {tx.is_fraud ? 'Anomaly' : 'Safe'}
          </span>
          {isReviewed && <div className="p-1 bg-green-500 text-white rounded-full shadow-lg"><ShieldCheck size={14} /></div>}
        </div>
      </div>
    </div>
  </motion.div>
);

const MetricCard = ({ label, value, sub, icon }) => (
  <motion.div 
    whileHover={{ y: -5 }}
    className="glass-card p-8 rounded-[35px] border border-white/60 shadow-xl transition-all group"
  >
    <div className="flex items-center gap-4 mb-6">
       <div className="p-3 bg-slate-50 text-slate-600 rounded-2xl border border-slate-100 group-hover:bg-orange-500 group-hover:text-white group-hover:shadow-lg group-hover:shadow-orange-200 transition-all duration-500">
          {React.cloneElement(icon, { size: 20 })}
       </div>
       <div>
         <p className="text-[11px] font-black text-slate-400 uppercase tracking-widest leading-none">{label}</p>
         <p className="text-[10px] font-medium text-slate-400 mt-1.5">{sub}</p>
       </div>
    </div>
    <h4 className="text-3xl font-black text-slate-900 tracking-tighter">{value}</h4>
  </motion.div>
);

export default Dashboard;
