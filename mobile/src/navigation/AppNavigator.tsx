/**
 * G_One_Sync AI — Mobile App Navigator
 * =====================================
 * Screen navigation orchestration handling auth state,
 * bottom tabs (Watchlist, Alerts), and patient detail views.
 */

import React, { useState } from 'react';
import { StyleSheet, Text, TouchableOpacity, View } from 'react-native';
import { AlertsScreen } from '../screens/AlertsScreen';
import { DashboardScreen } from '../screens/DashboardScreen';
import { LoginScreen } from '../screens/LoginScreen';
import { PatientDetailScreen } from '../screens/PatientDetailScreen';
import { clearSession, isSessionValid } from '../utils/auth';
import { THEME } from '../utils/constants';

type TabType = 'watchlist' | 'alerts';

export const AppNavigator: React.FC = () => {
  const [isAuthenticated, setIsAuthenticated] = useState<boolean>(() => isSessionValid());
  const [activeTab, setActiveTab] = useState<TabType>('watchlist');
  const [selectedPatientId, setSelectedPatientId] = useState<number | null>(null);

  const handleLoginSuccess = (_username: string) => {
    setIsAuthenticated(true);
    setSelectedPatientId(null);
  };

  const handleLogout = () => {
    clearSession();
    setIsAuthenticated(false);
    setSelectedPatientId(null);
  };

  if (!isAuthenticated) {
    return <LoginScreen onLoginSuccess={handleLoginSuccess} />;
  }

  // If a patient detail is open, show detail screen with back button
  if (selectedPatientId !== null) {
    return (
      <PatientDetailScreen
        patientId={selectedPatientId}
        onBack={() => setSelectedPatientId(null)}
      />
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.body}>
        {activeTab === 'watchlist' && (
          <DashboardScreen
            onSelectPatient={(id) => setSelectedPatientId(id)}
            onLogout={handleLogout}
          />
        )}
        {activeTab === 'alerts' && (
          <AlertsScreen
            onSelectPatient={(id) => setSelectedPatientId(id)}
          />
        )}
      </View>

      {/* Bottom Navigation Bar */}
      <View style={styles.bottomNav}>
        <TouchableOpacity
          style={[styles.navTab, activeTab === 'watchlist' && styles.navTabActive]}
          onPress={() => setActiveTab('watchlist')}
        >
          <Text style={[styles.navIcon, activeTab === 'watchlist' && styles.navIconActive]}>
            🩺
          </Text>
          <Text style={[styles.navLabel, activeTab === 'watchlist' && styles.navLabelActive]}>
            Watchlist
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[styles.navTab, activeTab === 'alerts' && styles.navTabActive]}
          onPress={() => setActiveTab('alerts')}
        >
          <Text style={[styles.navIcon, activeTab === 'alerts' && styles.navIconActive]}>
            🔔
          </Text>
          <Text style={[styles.navLabel, activeTab === 'alerts' && styles.navLabelActive]}>
            Alerts
          </Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: THEME.bgDark,
  },
  body: {
    flex: 1,
  },
  bottomNav: {
    flexDirection: 'row',
    backgroundColor: THEME.cardBg,
    borderTopWidth: 1,
    borderTopColor: THEME.cardBorder,
    paddingBottom: 24,
    paddingTop: 8,
  },
  navTab: {
    flex: 1,
    alignItems: 'center',
    paddingVertical: 6,
  },
  navTabActive: {},
  navIcon: {
    fontSize: 20,
    marginBottom: 4,
    opacity: 0.5,
  },
  navIconActive: {
    opacity: 1,
  },
  navLabel: {
    color: THEME.textMuted,
    fontSize: 11,
    fontWeight: '600',
  },
  navLabelActive: {
    color: THEME.primary,
    fontWeight: '800',
  },
});
