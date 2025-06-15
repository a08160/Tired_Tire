import 'dart:io';
import 'package:flutter/material.dart';
import 'home_page.dart';

class DiagnosisResultPage extends StatelessWidget {
  final Map<String, dynamic> result;
  final String imagePath;

  DiagnosisResultPage({required this.result, required this.imagePath});

  @override
  Widget build(BuildContext context) {
    double airPct = (result['air_pct'] ?? 0).toDouble();
    int score = airPct.round();

    // 상태 판정
    String statusText;
    Color statusColor;
    Color bgColor;
    String commentText;
    IconData statusIcon;

    if (score >= 80) {
      statusText = "양호";
      statusColor = Color(0xFF22C55E);
      bgColor = Color(0xFFE6F4E9);
      commentText = "공기압 상태가 양호해요!";
      statusIcon = Icons.verified;
    } else if (score >= 60) {
      statusText = "주의";
      statusColor = Color(0xFFFACC15);
      bgColor = Color(0xFFFFF7E0);
      commentText = "공기주입이 필요해요!";
      statusIcon = Icons.warning_amber_rounded;
    } else {
      statusText = "위험";
      statusColor = Color(0xFFEF4444);
      bgColor = Color(0xFFFFEBEB);
      commentText = "즉시 점검이 필요해요!";
      statusIcon = Icons.cancel;
    }

    return Scaffold(
      appBar: AppBar(
        title: Text("공기압 진단 결과"),
        backgroundColor: Colors.white,
        foregroundColor: Colors.black,
        elevation: 0,
      ),
      backgroundColor: Colors.white,
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            // 상태 뱃지
            Container(
              padding: EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              decoration: BoxDecoration(
                color: statusColor,
                borderRadius: BorderRadius.circular(30),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(statusIcon, color: Colors.white, size: 20),
                  SizedBox(width: 8),
                  Text(
                    statusText,
                    style: TextStyle(color: Colors.white, fontSize: 16),
                  ),
                ],
              ),
            ),
            SizedBox(height: 20),

            Text.rich(
              TextSpan(
                text: "공기압 점수 ",
                children: [
                  TextSpan(
                    text: "$score점",
                    style: TextStyle(
                      color: statusColor,
                      fontWeight: FontWeight.bold,
                      fontSize: 26,
                    ),
                  ),
                ],
              ),
              style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 10),
            Text(commentText, style: TextStyle(fontSize: 18)),
            SizedBox(height: 30),

            // 이미지 + 막대 시각화
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                _buildResultImage(),
                SizedBox(width: 20),
                _buildScoreBar(score, statusColor),
              ],
            ),

            SizedBox(height: 30),

            // AI 코멘트 박스
            Container(
              width: double.infinity,
              padding: EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: bgColor,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text("AI 코멘트", style: TextStyle(fontWeight: FontWeight.bold)),
                  SizedBox(height: 8),
                  Text(_generateAIComment(score)),
                ],
              ),
            ),
            SizedBox(height: 30),

            Column(
              children: [
                ElevatedButton.icon(
                  onPressed: () {
                    // TODO: 결과 저장 기능 나중에 구현
                    ScaffoldMessenger.of(
                      context,
                    ).showSnackBar(SnackBar(content: Text('저장 기능 준비 중')));
                  },
                  icon: Icon(Icons.save_alt),
                  label: Text("결과 저장"),
                  style: ElevatedButton.styleFrom(
                    minimumSize: Size(double.infinity, 50),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    backgroundColor: Colors.black,
                  ),
                ),
                SizedBox(height: 16),

                ElevatedButton.icon(
                  onPressed: () {
                    Navigator.pop(context);
                  },
                  icon: Icon(Icons.refresh),
                  label: Text("다시 진단하기"),
                  style: ElevatedButton.styleFrom(
                    minimumSize: Size(double.infinity, 50),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    backgroundColor: Colors.black,
                  ),
                ),
                SizedBox(height: 16),

                ElevatedButton.icon(
                  onPressed: () {
                    Navigator.pushAndRemoveUntil(
                      context,
                      MaterialPageRoute(builder: (context) => HomePage()),
                      (route) => false,
                    );
                  },
                  icon: Icon(Icons.home),
                  label: Text("홈으로"),
                  style: ElevatedButton.styleFrom(
                    minimumSize: Size(double.infinity, 50),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    backgroundColor: Colors.grey[400],
                  ),
                ),
              ],
            ),
          ], //children
        ),
      ),
    );
  }

  /// 이미지 표시 함수
  Widget _buildResultImage() {
    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: Image.file(
        File(imagePath),
        width: 180,
        height: 180,
        fit: BoxFit.cover,
      ),
    );
  }

  /// 막대 시각화
  Widget _buildScoreBar(int score, Color activeColor) {
    int activeIndex = _getScoreBarIndex(score);

    return Row(
      crossAxisAlignment: CrossAxisAlignment.center,
      children: [
        Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children:
              List.generate(10, (index) {
                bool isActive = (index == activeIndex);
                return Container(
                  margin: EdgeInsets.symmetric(vertical: 4),
                  alignment: Alignment.centerLeft,
                  child: Container(
                    width: isActive ? 60 : 40, // active 막대만 더 길게
                    height: 12,
                    decoration: BoxDecoration(
                      color: _getBarColor(index, activeIndex, activeColor),
                      borderRadius: BorderRadius.circular(6),
                    ),
                  ),
                );
              }).reversed.toList(),
        ),
        SizedBox(width: 12),
        Text(
          "$score점",
          style: TextStyle(
            color: activeColor,
            fontWeight: FontWeight.bold,
            fontSize: 20,
          ),
        ),
      ],
    );
  }

  /// 점수 구간 계산
  int _getScoreBarIndex(int score) {
    int index = (score / 10).floor();
    if (index >= 10) index = 9; // 100점도 마지막 막대로 포함
    return index;
  }

  /// 각 막대 색상 계산
  Color _getBarColor(int index, int activeIndex, Color activeColor) {
    if (index == activeIndex) return activeColor;
    if (index <= 5) return Color(0xFFFFCDD2); // 연한 빨강
    if (index <= 7) return Color(0xFFFFF9C4); // 연한 노랑
    return Color(0xFFC8E6C9); // 연한 초록
  }

  /// AI 코멘트
  String _generateAIComment(int score) {
    if (score >= 80) {
      return "타이어 공기압 점수가 ${score}점으로 양호한 상태입니다.";
    } else if (score >= 60) {
      return "타이어 공기압 점수가 ${score}점으로 공기주입이 필요합니다.";
    } else {
      return "타이어 공기압 점수가 ${score}점으로 위험합니다. 즉시 점검해주세요.";
    }
  }
}
