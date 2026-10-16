/**
 * G_One_Sync AI — Risk Badge Component
 * =====================================
 * Color-coded risk status badge with severity styling.
 */

import React from 'react';
import { StyleSheet, Text, View } from 'react-native';
import { RiskLevel } from '../types';
import { RISK_BG_COLORS, RISK_COLORS } from '../utils/constants';

interface RiskBadgeProps {
  level: RiskLevel;
  size?: 'small' | 'medium' | 'large';
  showScore?: boolean;
  score?: number;
}

export const RiskBadge: React.FC<RiskBadgeProps> = ({
  level,
  size = 'medium',
  showScore = false,
  score,
}) => {
  const color = RISK_COLORS[level] || RISK_COLORS.LOW;
  const bg = RISK_BG_COLORS[level] || RISK_BG_COLORS.LOW;

  const sizeStyles = {
    small: { paddingHorizontal: 8, paddingVertical: 2, fontSize: 11 },
    medium: { paddingHorizontal: 12, paddingVertical: 4, fontSize: 13 },
    large: { paddingHorizontal: 16, paddingVertical: 6, fontSize: 15 },
  }[size];

  return (
    <View
      style={[
        styles.container,
        {
          backgroundColor: bg,
          borderColor: color,
          paddingHorizontal: sizeStyles.paddingHorizontal,
          paddingVertical: sizeStyles.paddingVertical,
        },
      ]}
    >
      <View style={[styles.dot, { backgroundColor: color }]} />
      <Text style={[styles.text, { color, fontSize: sizeStyles.fontSize }]}>
        {level}
        {showScore && score !== undefined ? ` • ${score}` : ''}
      </Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    borderRadius: 9999,
    borderWidth: 1,
    alignSelf: 'flex-start',
  },
  dot: {
    width: 6,
    height: 6,
    borderRadius: 3,
    marginRight: 6,
  },
  text: {
    fontWeight: '700',
    letterSpacing: 0.5,
  },
});
