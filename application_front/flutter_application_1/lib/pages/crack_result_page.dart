import 'dart:io';
import 'package:flutter/material.dart';
import 'home_page.dart';
import 'package:flutter_application_1/services/diagnosis_service.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';

class CrackResultPage extends StatefulWidget {
  final Map<String, dynamic> result;
  final String userName;

  CrackResultPage({required this.result, required this.userName});

  @override
  State<CrackResultPage> createState() => _CrackResultPageState();
}

class _CrackResultPageState extends State<CrackResultPage> {
  void _showSaveDialog(
    BuildContext context,
    double score,
    String status,
    String comment,
    String imageUrl,
  ) async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;

    final carsSnapshot =
        await FirebaseFirestore.instance
            .collection('users')
            .doc(user.uid)
            .collection('cars')
            .get();

    final carDocs = carsSnapshot.docs;

    String? selectedCarId;
    String? selectedWheel;

    showDialog(
      context: context,
      builder: (context) {
        return StatefulBuilder(
          builder: (context, setState) {
            return AlertDialog(
              title: Text("차량과 타이어 위치 선택"),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // ✅ 드롭다운을 SizedBox로 감싸기
                  SizedBox(
                    width: 250,
                    child: DropdownButton<String>(
                      value: selectedCarId,
                      hint: Text("차량을 선택해주세요"),
                      isExpanded: true, // 내부 텍스트 줄바꿈 방지
                      items:
                          carDocs.map((doc) {
                            final model = doc['model'];
                            final plate = doc['plate'];
                            return DropdownMenuItem<String>(
                              value: doc.id,
                              child: Text("$model ($plate)"),
                            );
                          }).toList(),
                      onChanged: (value) {
                        setState(() {
                          selectedCarId = value;
                        });
                      },
                    ),
                  ),
                  SizedBox(height: 12),
                  SizedBox(
                    width: 250,
                    height: 250,
                    child: Stack(
                      alignment: Alignment.center,
                      children: [
                        // 차량 이미지
                        Image.asset('assets/car_top_view.png', width: 180),

                        // 좌측 앞바퀴
                        Positioned(
                          top: 50,
                          left: 10,
                          child: GestureDetector(
                            onTap:
                                () => setState(() => selectedWheel = "좌측 앞바퀴"),
                            child: Image.asset(
                              selectedWheel == "좌측 앞바퀴"
                                  ? 'assets/tire_blue.png'
                                  : 'assets/tire_black.png',
                              width: 50,
                            ),
                          ),
                        ),

                        // 우측 앞바퀴
                        Positioned(
                          top: 50,
                          right: 10,
                          child: GestureDetector(
                            onTap:
                                () => setState(() => selectedWheel = "우측 앞바퀴"),
                            child: Image.asset(
                              selectedWheel == "우측 앞바퀴"
                                  ? 'assets/tire_blue.png'
                                  : 'assets/tire_black.png',
                              width: 50,
                            ),
                          ),
                        ),

                        // 좌측 뒷바퀴
                        Positioned(
                          bottom: 50,
                          left: 10,
                          child: GestureDetector(
                            onTap:
                                () => setState(() => selectedWheel = "좌측 뒷바퀴"),
                            child: Image.asset(
                              selectedWheel == "좌측 뒷바퀴"
                                  ? 'assets/tire_blue.png'
                                  : 'assets/tire_black.png',
                              width: 50,
                            ),
                          ),
                        ),

                        // 우측 뒷바퀴
                        Positioned(
                          bottom: 50,
                          right: 10,
                          child: GestureDetector(
                            onTap:
                                () => setState(() => selectedWheel = "우측 뒷바퀴"),
                            child: Image.asset(
                              selectedWheel == "우측 뒷바퀴"
                                  ? 'assets/tire_blue.png'
                                  : 'assets/tire_black.png',
                              width: 50,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  if (selectedWheel != null)
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: Text("선택된 위치: $selectedWheel"),
                    ),
                ],
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: Text("취소"),
                ),
                ElevatedButton(
                  onPressed: () async {
                    if (selectedCarId == null || selectedWheel == null) {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(content: Text("차량과 위치를 모두 선택해주세요.")),
                      );
                      return;
                    }

                    await saveCrackDiagnosisResult(
                      carId: selectedCarId!,
                      riskScore: score,
                      status: status,
                      wheelPosition: selectedWheel!,
                      comment: comment,
                      imageUrl: imageUrl,
                    );

                    Navigator.pop(context);
                    ScaffoldMessenger.of(
                      context,
                    ).showSnackBar(SnackBar(content: Text("결과가 저장되었습니다.")));
                  },
                  child: Text("저장"),
                ),
              ],
            );
          },
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    double riskScore = (widget.result['risk_score'] ?? 0).toDouble();
    int score = riskScore.round();

    String statusText;
    Color statusColor;
    Color bgColor;
    String commentText;
    IconData statusIcon;

    if (score >= 70) {
      statusText = "양호";
      statusColor = Color(0xFF22C55E);
      bgColor = Color(0xFFE6F4E9);
      commentText = "균열이 거의 없어요!";
      statusIcon = Icons.verified;
    } else if (score >= 35) {
      statusText = "주의";
      statusColor = Color(0xFFFACC15);
      bgColor = Color(0xFFFFF7E0);
      commentText = "균열이 일부 보입니다.";
      statusIcon = Icons.warning_amber_rounded;
    } else {
      statusText = "위험";
      statusColor = Color(0xFFEF4444);
      bgColor = Color(0xFFFFEBEB);
      commentText = "균열이 심각합니다. 즉시 점검하세요!";
      statusIcon = Icons.cancel;
    }

    return Scaffold(
      appBar: AppBar(
        title: Text("균열 진단 결과"),
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
                text: "균열 위험도 점수 ",
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

            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                _buildLegendDot(Colors.red, '고위험도 균열'),
                SizedBox(width: 12),
                _buildLegendDot(Colors.orange, '중위험도 균열'),
                SizedBox(width: 12),
                _buildLegendDot(Colors.yellow[700]!, '저위험도 균열'),
              ],
            ),
            SizedBox(height: 30),

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
                    final double riskScore =
                        (widget.result['risk_score'] ?? 0).toDouble();
                    final int score = riskScore.round();
                    final String status =
                        score >= 70 ? "양호" : (score >= 35 ? "주의" : "위험");
                    final String comment = _generateAIComment(score);
                    final String imageUrl =
                        widget.result['blended_image_url'] ?? '';

                    _showSaveDialog(
                      context,
                      riskScore,
                      status,
                      comment,
                      imageUrl,
                    );
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
                      MaterialPageRoute(
                        builder:
                            (context) => HomePage(userName: widget.userName),
                      ),
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
          ],
        ),
      ),
    );
  }

  Widget _buildResultImage() {
    String imageUrl = widget.result['blended_image_url'] ?? '';

    return ClipRRect(
      borderRadius: BorderRadius.circular(16),
      child: Image.network(
        imageUrl,
        width: 180,
        height: 180,
        fit: BoxFit.cover,
        errorBuilder: (context, error, stackTrace) {
          return Container(
            width: 180,
            height: 180,
            color: Colors.grey[300],
            child: Icon(Icons.broken_image, size: 60, color: Colors.grey),
          );
        },
      ),
    );
  }

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
                    width: isActive ? 60 : 40,
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

  int _getScoreBarIndex(int score) {
    int index = (score / 10).floor();
    if (index >= 10) index = 9;
    return index;
  }

  Color _getBarColor(int index, int activeIndex, Color activeColor) {
    if (index == activeIndex) return activeColor;
    if (index <= 5) return Color(0xFFFFCDD2);
    if (index <= 7) return Color(0xFFFFF9C4);
    return Color(0xFFC8E6C9);
  }

  String _generateAIComment(int score) {
    if (score >= 70) {
      return "균열 위험도가 ${score}점으로 매우 양호한 상태입니다.";
    } else if (score >= 35) {
      return "균열 위험도가 ${score}점으로 약간의 균열이 발견되었습니다.";
    } else {
      return "균열 위험도가 ${score}점으로 심각한 균열이 있습니다. 즉시 점검이 필요합니다.";
    }
  }

  Widget _buildLegendDot(Color color, String label) {
    return Row(
      children: [
        Container(
          width: 12,
          height: 12,
          decoration: BoxDecoration(color: color, shape: BoxShape.circle),
        ),
        SizedBox(width: 4),
        Text(label, style: TextStyle(fontSize: 14)),
      ],
    );
  }
}
