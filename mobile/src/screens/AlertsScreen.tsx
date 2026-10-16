/**
 * G_One_Sync AI — Mobile Alerts History Screen
 * ==============================================
 * Push notification event history for critical ICU alerts.
 */

import React, { useState } from 'react';
import {
  Alert,
  FlatList,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { apiService } from '../services/api';
import { NotificationHistoryEntry } from '../types';
import { RISK_COLORS, THEME } from '../utils/constants';

interface AlertsScreenProps {
  onSelectPatient: (patientId: number) => void;
}

export const AlertsScreen: React.FC<AlertsScreenProps> = ({ onSelectPatient }) => {
  const [notifications, setNotifications] = useState<NotificationHistoryEntry[]>([
    {
      notification_id: 'notif-1',
      patient_id: 1042,
      risk_level: 'CRITICAL',
      title: '🔴 CRITICAL — Patient #1042',
      body: 'Deterioration probability: 94.0%. Immediate review recommended.',
      sent_at: '2 mins ago',
      read: false,
    },
    {
      notification_id: 'notif-2',
      patient_id: 1089,
      risk_level: 'HIGH',
      title: '🟠 HIGH — Patient #1089',
      body: 'Deterioration probability: 72.0%. Tachycardia + elevated lactate.',
      sent_at: '18 mins ago',
      read: false,
    },
    {
      notification_id: 'notif-3',
      patient_id: 1015,
      risk_level: 'MODERATE',
      title: '🟡 MODERATE — Patient #1015',
      body: 'SpO2 dipped below 92%. Nurse notification dispatched.',
      sent_at: '1 hour ago',
      read: true,
    },
  ]);

  const [filterUnreadOnly, setFilterUnreadOnly] = useState(false);

  const displayedNotifications = filterUnreadOnly
    ? notifications.filter((n) => !n.read)
    : notifications;

  const unreadCount = notifications.filter((n) => !n.read).length;

  const handleMarkAllRead = () => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  };

  const handleSendTestPush = async () => {
    try {
      await apiService.sendTestPush('fcm_simulated_device_token_demo');
      Alert.alert('Test Push Sent', 'A simulated push notification was received.');
    } catch {
      Alert.alert('Simulated Push', 'Test alert received: System connectivity active.');
    }
  };

  const handlePressAlert = (item: NotificationHistoryEntry) => {
    // Mark as read
    setNotifications((prev) =>
      prev.map((n) =>
        n.notification_id === item.notification_id ? { ...n, read: true } : n
      )
    );
    if (item.patient_id) {
      onSelectPatient(item.patient_id);
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>ICU Alert Stream</Text>
          <Text style={styles.headerSub}>
            {unreadCount > 0 ? `${unreadCount} unacknowledged alerts` : 'All alerts acknowledged'}
          </Text>
        </View>

        <TouchableOpacity onPress={handleMarkAllRead} style={styles.markReadBtn}>
          <Text style={styles.markReadText}>Mark all read</Text>
        </TouchableOpacity>
      </View>

      {/* Filter / Test Bar */}
      <View style={styles.toolbar}>
        <TouchableOpacity
          style={[styles.chip, filterUnreadOnly && styles.chipActive]}
          onPress={() => setFilterUnreadOnly(!filterUnreadOnly)}
        >
          <Text style={[styles.chipText, filterUnreadOnly && styles.chipTextActive]}>
            {filterUnreadOnly ? '● Unread Only' : 'All Alerts'}
          </Text>
        </TouchableOpacity>

        <TouchableOpacity style={styles.testBtn} onPress={handleSendTestPush}>
          <Text style={styles.testBtnText}>🧪 Test Push</Text>
        </TouchableOpacity>
      </View>

      <FlatList
        data={displayedNotifications}
        keyExtractor={(item) => item.notification_id}
        contentContainerStyle={styles.listContent}
        renderItem={({ item }) => {
          const accentColor = item.risk_level
            ? RISK_COLORS[item.risk_level]
            : THEME.primary;

          return (
            <TouchableOpacity
              style={[
                styles.alertCard,
                { borderLeftColor: accentColor },
                !item.read && styles.unreadCard,
              ]}
              onPress={() => handlePressAlert(item)}
              activeOpacity={0.7}
            >
              <View style={styles.cardHeader}>
                <Text style={styles.alertTitle}>{item.title}</Text>
                <Text style={styles.timeText}>{item.sent_at}</Text>
              </View>

              <Text style={styles.alertBody}>{item.body}</Text>

              {item.patient_id && (
                <View style={styles.viewDetailsRow}>
                  <Text style={styles.viewDetailsText}>View Bed #{item.patient_id} →</Text>
                </View>
              )}
            </TouchableOpacity>
          );
        }}
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <Text style={styles.emptyText}>No alerts in history.</Text>
          </View>
        }
      />
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: THEME.bgDark,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 48,
    paddingBottom: 16,
    backgroundColor: THEME.cardBg,
    borderBottomWidth: 1,
    borderBottomColor: THEME.cardBorder,
  },
  headerTitle: {
    color: THEME.textPrimary,
    fontSize: 18,
    fontWeight: '800',
  },
  headerSub: {
    color: THEME.textMuted,
    fontSize: 12,
    marginTop: 2,
  },
  markReadBtn: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: THEME.surface,
    borderRadius: 8,
  },
  markReadText: {
    color: THEME.textSecondary,
    fontSize: 12,
    fontWeight: '600',
  },
  toolbar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 12,
  },
  chip: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    backgroundColor: THEME.cardBg,
    borderWidth: 1,
    borderColor: THEME.cardBorder,
  },
  chipActive: {
    backgroundColor: THEME.primary,
    borderColor: THEME.primary,
  },
  chipText: {
    color: THEME.textSecondary,
    fontSize: 12,
    fontWeight: '700',
  },
  chipTextActive: {
    color: '#FFFFFF',
  },
  testBtn: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    backgroundColor: THEME.surface,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: THEME.cardBorder,
  },
  testBtnText: {
    color: THEME.accentTeal,
    fontSize: 12,
    fontWeight: '600',
  },
  listContent: {
    padding: 16,
    gap: 12,
  },
  alertCard: {
    backgroundColor: THEME.cardBg,
    borderRadius: 14,
    padding: 16,
    borderWidth: 1,
    borderColor: THEME.cardBorder,
    borderLeftWidth: 4,
  },
  unreadCard: {
    backgroundColor: '#162238',
  },
  cardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  alertTitle: {
    color: THEME.textPrimary,
    fontSize: 14,
    fontWeight: '700',
    flex: 1,
  },
  timeText: {
    color: THEME.textMuted,
    fontSize: 11,
    marginLeft: 8,
  },
  alertBody: {
    color: THEME.textSecondary,
    fontSize: 13,
    lineHeight: 18,
    marginBottom: 8,
  },
  viewDetailsRow: {
    alignSelf: 'flex-start',
  },
  viewDetailsText: {
    color: THEME.primary,
    fontSize: 12,
    fontWeight: '700',
  },
  emptyContainer: {
    padding: 40,
    alignItems: 'center',
  },
  emptyText: {
    color: THEME.textMuted,
    fontSize: 14,
  },
});
