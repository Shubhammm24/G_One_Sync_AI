/**
 * G_One_Sync AI — Vital Card Component
 * =====================================
 * Compact telemetry card showing clinical vital value,
 * units, normal bounds, and abnormal status indicator.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { THEME } from '../utils/constants';

interface VitalCardProps {
  label: string;
  value: number;
  unit: string;
  min: number;
  max: number;
  formatDecimals?: number;
}

export const VitalCard: React.FC<VitalCardProps> = ({
  label,
  value,
  unit,
  min,
  max,
  formatDecimals = 0,
}) => {
  const isAbnormalLow = value < min;
  const isAbnormalHigh = value > max;
  const isAbnormal = isAbnormalLow || isAbnormalHigh;

  const statusColor = isAbnormal ? '#EF4444' : '#10B981';
  const statusBg = isAbnormal ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.05)';

  return (
    <View style={[styles.card, { borderColor: isAbnormal ? '#EF444488' : THEME.cardBorder, backgroundColor: statusBg }]}>
      <View style={styles.header}>
        <Text style={styles.label}>{label}</Text>
        {isAbnormal && (
          <View style={[styles.badge, { backgroundColor: '#EF444422' }]}>
            <Text style={styles.badgeText}>{isAbnormalLow ? 'LOW' : 'HIGH'}</Text>
          </View>
        )}
      </View>

      <View style={styles.valueRow}>
        <Text style={[styles.value, { color: statusColor }]}>
          {formatDecimals > 0 ? value.toFixed(formatDecimals) : Math.round(value)}
        </Text>
        <Text style={styles.unit}>{unit}</Text>
      </View>

      <Text style={styles.range}>
        Normal: {min} - {max}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    flex: 1,
    minWidth: 100,
    padding: 12,
    borderRadius: 12,
    borderWidth: 1,
    margin: 4,
    backgroundColor: THEME.cardBg,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 6,
  },
  label: {
    color: THEME.textSecondary,
    fontSize: 12,
    fontWeight: '600',
    textTransform: 'uppercase',
  },
  badge: {
    paddingHorizontal: 5,
    paddingVertical: 1,
    borderRadius: 4,
  },
  badgeText: {
    color: '#EF4444',
    fontSize: 9,
    fontWeight: '700',
  },
  valueRow: {
    flexDirection: 'row',
    alignItems: 'baseline',
    marginBottom: 4,
  },
  value: {
    fontSize: 22,
    fontWeight: '800',
    letterSpacing: -0.5,
  },
  unit: {
    color: THEME.textMuted,
    fontSize: 11,
    marginLeft: 4,
    fontWeight: '500',
  },
  range: {
    color: THEME.textMuted,
    fontSize: 10,
  },
});
