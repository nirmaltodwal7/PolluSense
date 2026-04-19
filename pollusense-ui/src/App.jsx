/**
 * App.jsx
 * ------
 * Root component for the PolluSense React application.
 */

import React from 'react';
import Dashboard from './components/Dashboard.jsx';
import './App.css';

function App() {
  return (
    <div className="App">
      <Dashboard />
    </div>
  );
}

export default App;
