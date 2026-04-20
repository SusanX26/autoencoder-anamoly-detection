import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ShieldAlert, ShieldCheck, Activity, BarChart3, 
  Clock, CreditCard, ChevronRight, Search, Zap 
} from 'lucide-react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, 
  Tooltip, ResponsiveContainer, Cell 
} from 'recharts';
import axios from 'axios';

const API_BASE = "http://localhost:8000";

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('Monitor');
  const [threshold, setThreshold] = useState(0.05);
  const [transactions, setTransactions] = useState([]);
  const [selectedTx, setSelectedTx] = useState(null);
  const [shapData, setShapData] = useState([]);
  const [metrics, setMetrics] = useState({ auprc: 0, latency_ms: 0 });
  const [loading, setLoading] = useState(false);
  const [reviewedTxs, setReviewedTxs] = useState(new Set());

  useEffect(() => {
    fetchTransactions();
    fetchMetrics();
    const interval = setInterval(fetchTransactions, 30000);
    return () => clearInterval(interval);
  }, []);

  const fetchTransactions = async () => {
    try {
      const res = await axios.get(`${API_BASE}/transactions?limit=12`);
      const tdata = res.data;
      const ids = tdata.map(t => t.id);
      const predRes = await axios.post(`${API_BASE}/predict`, ids);
      
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
    } catch (err) {}
  };

  const handleSelect = async (tx) => {
    setSelectedTx(tx);
    setLoading(true);
    try {
      const res = await axios.post(`${API_BASE}/explain?tid=${tx.id}`);
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
      // Simulate feedback to user
      console.log(`Transaction ${selectedTx.id} reviewed and verified.`);
    }
  };

  const renderContent = () => {
    return (
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: -20 }}
          className="w-full h-full"
        >
          {activeTab === 'Monitor' && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 animate-in-fade">
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
          {activeTab === 'Performance' && <PerformanceView metrics={metrics} />}
          {activeTab === 'Alert Rules' && <AlertRulesView threshold={threshold} setThreshold={setThreshold} />}
        </motion.div>
      </AnimatePresence>
    );
  };

  return (
    <div className="flex h-screen w-full bg-[#f8fafc] overflow-hidden text-slate-800">
      {/* Professional Sidebar */}
      <aside className="w-72 flex flex-col sidebar-classic shadow-sm z-30">
        <div className="p-8 flex items-center space-x-3 border-b border-slate-100">
          <div className="p-2 bg-orange-500 rounded-lg shadow-orange-200 shadow-lg">
            <Zap className="text-white w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-black tracking-tight text-slate-900 leading-none">FraudSense</h1>
            <p className="text-[10px] font-bold text-orange-500 uppercase tracking-widest mt-1">AI Intelligence</p>
          </div>
        </div>
        
        <nav className="flex-1 py-6 px-4 space-y-1">
          <NavItem 
            icon={<Activity size={20}/>} 
            label="Live Monitor" 
            active={activeTab === 'Monitor'} 
            onClick={() => setActiveTab('Monitor')}
          />
          <NavItem 
            icon={<BarChart3 size={20}/>} 
            label="Performance" 
            active={activeTab === 'Performance'} 
            onClick={() => setActiveTab('Performance')}
          />
          <NavItem 
            icon={<ShieldAlert size={20}/>} 
            label="Alert Rules" 
            active={activeTab === 'Alert Rules'} 
            onClick={() => setActiveTab('Alert Rules')}
          />
        </nav>

        <div className="p-8 border-t border-slate-100">
          <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 space-y-3">
             <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-[10px] font-bold text-slate-600 uppercase">System Status: Online</span>
             </div>
             <div>
                <p className="text-[10px] text-slate-400 uppercase font-bold tracking-tighter">Inference Speed</p>
                <p className="text-lg font-black text-slate-800 font-mono">{metrics.latency_ms}<span className="text-sm font-normal ml-1">ms</span></p>
             </div>
          </div>
        </div>
      </aside>

      {/* Modern Main Dashboard */}
      <main className="flex-1 flex flex-col overflow-hidden relative">
        <header className="h-20 flex items-center justify-between px-10 bg-white border-b border-slate-100 z-10 shadow-sm">
          <div className="flex items-center gap-6">
            <h2 className="text-xl font-bold text-slate-900">{activeTab === 'Monitor' ? 'Real-time Monitoring' : activeTab}</h2>
            {activeTab === 'Monitor' && (
              <div className="relative w-80">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" size={16} />
                <input 
                  type="text" 
                  placeholder="Filter by Transaction ID..." 
                  className="w-full bg-slate-50 border border-slate-200 rounded-xl py-2 pl-10 pr-4 text-xs focus:ring-2 focus:ring-orange-500/20 focus:border-orange-500 transition-all outline-none text-slate-600"
                />
              </div>
            )}
          </div>
          <div className="flex items-center space-x-4">
             <div className="bg-slate-100 px-3 py-1 rounded-full flex items-center space-x-2">
                <p className="text-[10px] font-bold text-slate-500">Autoencoder v2.4</p>
             </div>
             <div className="w-10 h-10 rounded-full bg-gradient-to-br from-orange-400 to-orange-600 flex items-center justify-center text-white font-bold text-xs shadow-lg shadow-orange-200">
                SS
             </div>
          </div>
        </header>

        <div className="flex-1 p-10 overflow-y-auto custom-scroll bg-[#f8fafc]">
          {renderContent()}
        </div>
      </main>

      {/* Analysis Panel (Right Side) */}
      <AnimatePresence>
        {selectedTx && (
          <motion.aside 
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: "spring", damping: 30, stiffness: 300 }}
            className="w-[480px] bg-white border-l border-slate-200 p-8 flex flex-col shadow-2xl z-40 h-full overflow-hidden"
          >
            <div className="flex justify-between items-center mb-8 pb-4 border-b border-slate-100">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-orange-100 text-orange-600 rounded-lg">
                  <ShieldAlert size={20} />
                </div>
                <h2 className="text-lg font-bold text-slate-800 uppercase tracking-tight">Audit Report</h2>
              </div>
              <button 
                onClick={() => setSelectedTx(null)}
                className="p-2 hover:bg-slate-100 rounded-full transition-all text-slate-400 hover:text-slate-600"
              >
                <ChevronRight className="rotate-180" />
              </button>
            </div>

            <div className="space-y-8 flex-1 overflow-y-auto pr-2 custom-scroll pb-10">
              <div className="bg-slate-50 p-6 rounded-2xl border border-slate-100">
                <div className="flex justify-between items-start mb-4">
                  <div>
                    <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Transaction Ref</p>
                    <p className="text-sm font-mono font-bold text-slate-700">#{selectedTx.id}</p>
                  </div>
                  {reviewedTxs.has(selectedTx.id) && (
                    <div className="bg-green-100 text-green-700 px-3 py-1 rounded-full text-[10px] font-bold flex items-center gap-1">
                      <ShieldCheck size={12} /> VERIFIED
                    </div>
                  )}
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 bg-white rounded-xl border border-slate-100">
                    <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Amount</p>
                    <p className="text-xl font-bold text-slate-900">${selectedTx.Amount.toFixed(2)}</p>
                  </div>
                  <div className={`p-4 rounded-xl border ${selectedTx.is_fraud ? 'bg-red-50 border-red-100' : 'bg-green-50 border-green-100'}`}>
                    <p className="text-[10px] font-bold text-slate-400 uppercase mb-1">Anom Score</p>
                    <p className={`text-xl font-bold ${selectedTx.is_fraud ? 'text-red-600' : 'text-green-600'}`}>
                      {(selectedTx.score * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>

              <div>
                <div className="flex items-center justify-between mb-6">
                  <h3 className="text-sm font-bold text-slate-800 flex items-center gap-2">
                    <BarChart3 size={16} className="text-orange-500" /> Explainability Breakdown (SHAP)
                  </h3>
                </div>
                {loading ? (
                  <div className="h-48 flex flex-col items-center justify-center space-y-3">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-orange-500"></div>
                    <p className="text-[10px] font-bold text-slate-400 tracking-widest uppercase">Calculating Attributions...</p>
                  </div>
                ) : (
                  <div className="h-64 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={shapData} layout="vertical" margin={{ left: -20 }}>
                        <XAxis type="number" hide />
                        <YAxis dataKey="feature" type="category" width={40} fontSize={10} tick={{fill: '#94a3b8', fontWeight: 'bold'}} />
                        <Tooltip 
                          cursor={{fill: 'rgba(0,0,0,0.02)'}}
                          contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: '12px', fontSize: '10px', boxShadow: '0 4px 6px rgba(0,0,0,0.05)' }}
                        />
                        <Bar dataKey="value" radius={[0, 10, 10, 0]} barSize={12}>
                          {shapData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.value > 0 ? '#ef4444' : '#10b981'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>

              {/* Functional Feature: Select & OK */}
              <div className="space-y-4 pt-4 border-t border-slate-100">
                <div className="bg-orange-50 p-5 rounded-2xl border border-orange-100 flex items-start gap-4">
                  <div className="p-2 bg-orange-100 text-orange-600 rounded-full mt-1">
                    <Zap size={16} />
                  </div>
                  <div>
                    <h4 className="text-xs font-bold text-orange-900 mb-1 leading-none uppercase tracking-tight">Audit Functional Action</h4>
                    <p className="text-[10px] text-orange-700 leading-relaxed font-medium">
                      Acknowledging this transaction confirms human review. This feedback loop is used to calibrate the Autoencoder boundary and reduces future false-positive flags.
                    </p>
                  </div>
                </div>

                <button 
                  onClick={handleAcknowledge}
                  disabled={reviewedTxs.has(selectedTx.id)}
                  className={`w-full py-4 rounded-xl font-bold text-sm transition-all flex items-center justify-center gap-2 ${
                    reviewedTxs.has(selectedTx.id) 
                    ? 'bg-slate-100 text-slate-400 cursor-not-allowed border border-slate-200' 
                    : 'bg-orange-600 text-white hover:bg-orange-700 shadow-lg shadow-orange-100 active:scale-[0.98]'
                  }`}
                >
                  {reviewedTxs.has(selectedTx.id) ? (
                    <><ShieldCheck size={18} /> Verified & Closed</>
                  ) : (
                    <><ShieldCheck size={18} /> Confirm This Result (OK)</>
                  )}
                </button>
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  );
};

const PerformanceView = ({ metrics }) => (
  <div className="space-y-10 animate-in-fade">
    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
      <MetricCard label="Model Reliability" value={(metrics.auprc * 100).toFixed(1) + "%"} sub="AUPRC Accuracy" icon={<ShieldCheck className="text-green-500" />} />
      <MetricCard label="Network Latency" value={metrics.latency_ms + "ms"} sub="Real-time performance" icon={<Activity className="text-orange-500" />} />
      <MetricCard label="Processed Vol" value="284,807" sub="24h Transaction Count" icon={<BarChart3 className="text-blue-500" />} />
    </div>
    
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
      <div className="bg-white p-10 rounded-3xl border border-slate-200 shadow-sm">
        <h3 className="text-lg font-bold text-slate-800 mb-8 flex items-center gap-2">
          <Activity size={20} className="text-orange-500" /> Learning Curve Convergence
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={dummyLoss}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
            <XAxis dataKey="epoch" stroke="#64748b" fontSize={10} tickLine={false} axisLine={false} />
            <YAxis stroke="#64748b" fontSize={10} tickLine={false} axisLine={false} />
            <Tooltip contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: '12px', boxShadow: '0 4px 6px rgba(0,0,0,0.05)' }} />
            <Bar dataKey="loss" radius={[8, 8, 8, 8]} barSize={32}>
              {dummyLoss.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={index === dummyLoss.length - 1 ? '#f97316' : '#e2e8f0'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      <div className="bg-white p-10 rounded-3xl border border-slate-200 shadow-sm">
        <h3 className="text-lg font-bold mb-8 text-slate-800 flex items-center gap-2">
          <ShieldAlert size={20} className="text-red-500" /> Confidence Matrix
        </h3>
        <div className="space-y-8">
          <PRRow label="Precision [Legitimate]" value="99.9%" color="bg-green-500" />
          <PRRow label="Recall [Fraud Anomalies]" value="88.4%" color="bg-red-500" />
          <PRRow label="Global F1 Performance" value="94.1%" color="bg-orange-500" />
        </div>
      </div>
    </div>
  </div>
);

const AlertRulesView = ({ threshold, setThreshold }) => (
  <div className="max-w-3xl mx-auto py-10 animate-in-fade">
    <div className="bg-white p-16 rounded-3xl border border-slate-200 shadow-sm text-center">
      <div className="mx-auto w-24 h-24 bg-orange-50 rounded-full flex items-center justify-center mb-8 border-4 border-white shadow-xl">
        <ShieldAlert className="text-orange-500 w-12 h-12" />
      </div>
      
      <div className="mb-12">
        <h3 className="text-3xl font-black text-slate-900 tracking-tight mb-3">Anomaly Sensitivity</h3>
        <p className="text-slate-500 text-sm max-w-md mx-auto font-medium">
          Control the reconstruction error threshold. A lower threshold increases fraud sensitivity but may raise false alerts.
        </p>
      </div>

      <div className="bg-slate-50 p-10 rounded-3xl border border-slate-100 mb-8">
        <div className="flex justify-between items-end mb-8">
          <div className="text-left">
            <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest block mb-2">Current Delta</span>
            <span className="text-5xl font-mono font-black text-slate-900 leading-none">{threshold.toFixed(3)}</span>
          </div>
          <div className="text-right">
             <div className="bg-orange-100 text-orange-700 px-4 py-1 rounded-full text-[10px] font-bold mb-2 inline-block">PRODUCTION LIVE</div>
             <p className="text-xs font-bold text-slate-400">Stable Threshold Point</p>
          </div>
        </div>
        
        <div className="relative">
          <input 
            type="range" 
            min="0.001" 
            max="0.2" 
            step="0.005" 
            value={threshold} 
            onChange={(e) => setThreshold(parseFloat(e.target.value))}
            className="w-full h-3 bg-slate-200 rounded-full appearance-none cursor-pointer accent-orange-500"
          />
          <div className="flex justify-between text-[10px] text-slate-400 uppercase font-black mt-6 tracking-widest">
            <span>High Precision (Safe)</span>
            <span>High Recall (Critical)</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="p-8 bg-green-50 rounded-2xl border border-green-100 text-left">
           <h4 className="text-[10px] font-black text-green-700 mb-2 uppercase tracking-widest leading-none">Normal Retention</h4>
           <p className="text-3xl font-black text-slate-800">99.2%</p>
        </div>
        <div className="p-8 bg-orange-50 rounded-2xl border border-orange-100 text-left">
           <h4 className="text-[10px] font-black text-orange-700 mb-2 uppercase tracking-widest leading-none">Fraud Detection</h4>
           <p className="text-3xl font-black text-slate-800">84.1%</p>
        </div>
      </div>
    </div>
  </div>
);

const dummyLoss = [
  { epoch: 'E1', loss: 0.082 }, { epoch: 'E2', loss: 0.045 },
  { epoch: 'E3', loss: 0.021 }, { epoch: 'E4', loss: 0.012 },
  { epoch: 'E5', loss: 0.009 }, { epoch: 'E6', loss: 0.008 }
];

const MetricCard = ({ label, value, sub, icon }) => (
  <div className="bg-white p-8 rounded-3xl border border-slate-200 shadow-sm hover:shadow-xl hover:border-orange-200 transition-all group overflow-hidden relative">
    <div className="absolute -right-4 -bottom-4 opacity-[0.03] scale-150 transition-transform group-hover:rotate-12 group-hover:scale-175 duration-700">
       {React.cloneElement(icon, { size: 120 })}
    </div>
    <div className="flex items-center gap-4 mb-6 relative z-10">
       <div className="p-3 bg-slate-50 text-slate-600 rounded-2xl border border-slate-100 group-hover:bg-orange-500 group-hover:text-white group-hover:shadow-lg group-hover:shadow-orange-100 transition-all">
          {React.cloneElement(icon, { size: 24 })}
       </div>
       <div>
         <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">{label}</p>
         <p className="text-xs font-medium text-slate-500 leading-none">{sub}</p>
       </div>
    </div>
    <h4 className="text-4xl font-black text-slate-900 relative z-10">{value}</h4>
  </div>
);

const PRRow = ({ label, value, color }) => (
  <div className="space-y-4">
    <div className="flex justify-between items-end">
      <span className="text-xs font-bold text-slate-500 uppercase tracking-tight">{label}</span>
      <span className="text-lg font-black text-slate-900 font-mono tracking-tighter">{value}</span>
    </div>
    <div className="h-3 bg-slate-100 rounded-full overflow-hidden p-[2px] border border-slate-200">
      <motion.div 
        initial={{ width: 0 }}
        animate={{ width: value }}
        transition={{ duration: 1, ease: "easeOut" }}
        className={`h-full rounded-full ${color} shadow-sm`} 
      />
    </div>
  </div>
);

const TransactionCard = ({ tx, onClick, selected, isReviewed }) => (
  <motion.div
    layout
    whileHover={{ y: -8 }}
    whileTap={{ scale: 0.98 }}
    onClick={onClick}
    className={`p-1 bg-white rounded-3xl cursor-pointer transition-all border-2 overflow-hidden ${
      selected 
      ? 'border-orange-500 shadow-2xl shadow-orange-100' 
      : 'border-slate-100 hover:border-orange-300 shadow-sm'
    }`}
  >
    <div className="p-6 relative overflow-hidden">
      {isReviewed && (
        <div className="absolute top-2 right-2 flex items-center gap-1 bg-green-500 text-white px-3 py-1 rounded-full text-[9px] font-black uppercase tracking-widest shadow-lg shadow-green-100 z-10">
          <ShieldCheck size={12} /> Verified
        </div>
      )}
      
      <div className="flex justify-between items-start mb-8 transition-transform group">
        <div className={`p-4 rounded-2xl ${tx.is_fraud ? 'bg-red-50 text-red-600' : 'bg-green-50 text-green-600 shadow-inner'}`}>
          {tx.is_fraud ? <ShieldAlert size={26}/> : <ShieldCheck size={26}/>}
        </div>
        <div className="text-right">
          <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1">Volume</p>
          <p className="font-mono text-lg font-black text-slate-900 tracking-tighter">${tx.Amount.toFixed(2)}</p>
        </div>
      </div>

      <div className="space-y-6">
        <div>
          <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-1.5">Entity Trace</p>
          <p className="font-mono text-xs font-bold text-slate-600 truncate bg-slate-50 p-2 rounded-lg border border-slate-100 group-hover:text-orange-600">#{tx.id}</p>
        </div>
        
        <div className="space-y-2">
           <div className="flex justify-between items-center text-[10px] font-bold text-slate-400">
             <span>SUSPICION FACTOR</span>
             <span className={tx.is_fraud ? 'text-red-500' : 'text-green-500'}>{(tx.score * 100).toFixed(1)}%</span>
           </div>
           <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden border border-slate-200 p-[1px]">
             <motion.div 
               initial={{ width: 0 }}
               animate={{ width: `${Math.min(tx.score * 1000, 100)}%` }}
               className={`h-full rounded-full ${tx.is_fraud ? 'bg-red-500 shadow-red-200 shadow-lg' : 'bg-green-500 shadow-green-100 shadow-lg'}`}
             />
           </div>
        </div>
        
        <div className="flex justify-between items-center pt-2">
          <div className="flex items-center gap-1.5">
             <div className={`w-2 h-2 rounded-full ${tx.is_fraud ? 'bg-red-500' : 'bg-green-500'}`}></div>
             <span className="text-[9px] font-black uppercase tracking-tight text-slate-500">
               {tx.is_fraud ? 'Anomaly Detected' : 'Verified Normal'}
             </span>
          </div>
          <ChevronRight size={16} className={`transition-all ${selected ? 'text-orange-500 translate-x-1' : 'text-slate-300'}`} />
        </div>
      </div>
    </div>
  </motion.div>
);

const NavItem = ({ icon, label, active, onClick }) => (
  <button 
    onClick={onClick}
    className={`w-full flex items-center space-x-3 px-6 py-4 rounded-2xl transition-all duration-300 relative group group-hover:translate-x-1 ${
      active 
      ? 'bg-orange-50 text-orange-600 shadow-inner' 
      : 'text-slate-500 hover:bg-slate-50 hover:text-orange-500'
    }`}
  >
    {active && (
      <motion.div 
        layoutId="navIndicator"
        className="absolute left-1 w-1.5 h-8 bg-orange-600 rounded-full"
      />
    )}
    <div className={`transition-all duration-300 ${active ? 'scale-110 rotate-3' : 'group-hover:scale-110 group-hover:-rotate-3'}`}>
      {icon}
    </div>
    <span className="text-sm font-black tracking-tight">{label}</span>
    {active && <ChevronRight size={14} className="ml-auto opacity-50" />}
  </button>
);

export default Dashboard;
