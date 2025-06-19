import 'package:flutter/material.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

class DiagnosisRecord {
  final String type; // 'air' or 'crack'
  final String id;
  final DateTime createdAt;
  final String wheelPosition;
  final int score;
  final String status;
  final String comment;

  DiagnosisRecord({
    required this.type,
    required this.id,
    required this.createdAt,
    required this.wheelPosition,
    required this.score,
    required this.status,
    required this.comment,
  });
}

class TireHistoryPage extends StatefulWidget {
  final String? selectedCarId; // ← 추가
  const TireHistoryPage({Key? key, this.selectedCarId}) : super(key: key);

  @override
  _TireHistoryPageState createState() => _TireHistoryPageState();
}

class _TireHistoryPageState extends State<TireHistoryPage> {
  String? nickname;
  List<Map<String, dynamic>> cars = [];
  String? selectedCarId;
  bool sortDescending = true;
  bool isLoading = false;
  String filterType = 'all'; // all, air, crack

  List<DiagnosisRecord> records = [];
  Set<String> expandedIds = {}; // record.id를 저장

  Set<String> selectedIds = {};
  bool selectMode = false;
  @override
  void initState() {
    super.initState();
    _loadUserAndCars().then((_) {
      if (widget.selectedCarId != null) {
        selectedCarId = widget.selectedCarId; // 💡 먼저 직접 설정
        _loadDiagnosisRecords(); // 💡 진단내역 바로 불러오기
      }
    });
  }

  void _confirmDeleteDialog() {
    showDialog(
      context: context,
      builder:
          (context) => AlertDialog(
            title: Text("삭제 확인"),
            content: Text("선택한 진단 내역을 삭제하시겠습니까?"),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(context),
                child: Text("취소"),
              ),
              TextButton(
                onPressed: () async {
                  Navigator.pop(context);
                  await _deleteSelectedRecords();
                },
                child: Text("삭제", style: TextStyle(color: Colors.red)),
              ),
            ],
          ),
    );
  }

  Future<void> _loadDiagnosisRecords() async {
    if (selectedCarId == null) return;
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;

    setState(() => isLoading = true); // 로딩 시작

    final airSnapshot =
        await FirebaseFirestore.instance
            .collection('users')
            .doc(user.uid)
            .collection('cars')
            .doc(selectedCarId)
            .collection('air')
            .get();

    final crackSnapshot =
        await FirebaseFirestore.instance
            .collection('users')
            .doc(user.uid)
            .collection('cars')
            .doc(selectedCarId)
            .collection('crack')
            .get();

    List<DiagnosisRecord> loaded = [];

    for (var doc in airSnapshot.docs) {
      final data = doc.data();
      final timestamp = data['createdAt'] as Timestamp?;
      if (timestamp == null) continue;
      loaded.add(
        DiagnosisRecord(
          type: 'air',
          id: doc.id,
          createdAt: timestamp.toDate(),
          wheelPosition: data['wheelPosition'] ?? '',
          score: (data['air_pct'] ?? 0).round(),
          status: data['status'] ?? '',
          comment: data['comment'] ?? '',
        ),
      );
    }

    for (var doc in crackSnapshot.docs) {
      final data = doc.data();
      final timestamp = data['createdAt'] as Timestamp?;
      if (timestamp == null) continue;
      loaded.add(
        DiagnosisRecord(
          type: 'crack',
          id: doc.id,
          createdAt: timestamp.toDate(),
          wheelPosition: data['wheelPosition'] ?? '',
          score: (data['risk_score'] ?? 0).round(),
          status: data['status'] ?? '',
          comment: data['comment'] ?? '',
        ),
      );
    }
    loaded =
        loaded
            .where((r) => filterType == 'all' || r.type == filterType)
            .toList();
    loaded.sort(
      (a, b) =>
          sortDescending
              ? b.createdAt.compareTo(a.createdAt)
              : a.createdAt.compareTo(b.createdAt),
    );

    setState(() {
      records = loaded;
      isLoading = false; // 로딩 끝
    });
  }

  Future<void> _loadUserAndCars() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null) return;

    final userDoc =
        await FirebaseFirestore.instance
            .collection('users')
            .doc(user.uid)
            .get();
    final carSnapshot =
        await FirebaseFirestore.instance
            .collection('users')
            .doc(user.uid)
            .collection('cars')
            .get();

    setState(() {
      nickname = userDoc.data()?['nickname'] ?? '';
      cars =
          carSnapshot.docs.map((doc) {
            return {'id': doc.id, 'model': doc['model'], 'plate': doc['plate']};
          }).toList();
      if (cars.isNotEmpty) selectedCarId = cars[0]['id'];
    });
  }

  Future<void> _deleteSelectedRecords() async {
    final user = FirebaseAuth.instance.currentUser;
    if (user == null || selectedCarId == null) return;

    final batch = FirebaseFirestore.instance.batch();

    for (final id in selectedIds) {
      final record = records.firstWhere((r) => r.id == id);
      final docRef = FirebaseFirestore.instance
          .collection('users')
          .doc(user.uid)
          .collection('cars')
          .doc(selectedCarId)
          .collection(record.type)
          .doc(id);

      batch.delete(docRef);
    }

    await batch.commit();
    setState(() {
      selectedIds.clear();
      selectMode = false;
    });
    await _loadDiagnosisRecords(); // 리스트 갱신
  }

  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: () async {
        Navigator.pop(context, selectedCarId); // 선택된 차량 ID를 HomePage로 전달
        return false; // 시스템 뒤로가기 막고 직접 pop 처리
      },
      child: Scaffold(
        appBar: AppBar(
          title: Text("진단 내역"),
          actions: [
            if (records.isNotEmpty)
              IconButton(
                icon: Icon(selectMode ? Icons.cancel : Icons.check_box),
                onPressed: () {
                  setState(() {
                    selectMode = !selectMode;
                    selectedIds.clear();
                  });
                },
              ),
          ],
        ),
        body: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (nickname != null)
                Text(
                  "$nickname 님의 진단내역",
                  style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
              SizedBox(height: 12),
              _buildCarDropdown(),
              SizedBox(height: 12),
              _buildSortAndFilter(),
              Expanded(child: _buildDiagnosisList()),
            ],
          ),
        ),
        floatingActionButton:
            selectMode && selectedIds.isNotEmpty
                ? FloatingActionButton(
                  onPressed: () => _confirmDeleteDialog(),
                  backgroundColor: Colors.red,
                  child: Icon(Icons.delete),
                )
                : null,
      ),
    );
  }

  Widget _buildCarDropdown() {
    return DropdownButton<String>(
      value: selectedCarId,
      isExpanded: true,
      hint: Text("차량을 선택해주세요"),
      items:
          cars.map((car) {
            final label = "${car['model']} (${car['plate']})";
            return DropdownMenuItem<String>(
              value: car['id'],
              child: Text(label),
            );
          }).toList(),
      onChanged: (value) async {
        setState(() {
          selectedCarId = value;
          isLoading = true;
          records = []; // 변경 시 잠깐 리스트 비우기
        });
        await _loadDiagnosisRecords(); // 차량 변경 즉시 데이터 로드
      },
    );
  }

  Widget _buildSortAndFilter() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Row(
          children: [
            TextButton.icon(
              onPressed: () {
                setState(() => sortDescending = !sortDescending);
                _loadDiagnosisRecords();
              },
              icon: Icon(
                sortDescending ? Icons.arrow_downward : Icons.arrow_upward,
              ),
              label: Text("날짜"),
            ),
            SizedBox(width: 8),
            DropdownButton<String>(
              value: filterType,
              items: [
                DropdownMenuItem(value: 'all', child: Text('전체')),
                DropdownMenuItem(value: 'air', child: Text('공기압')),
                DropdownMenuItem(value: 'crack', child: Text('균열')),
              ],
              onChanged: (value) {
                if (value != null) {
                  setState(() => filterType = value);
                  _loadDiagnosisRecords();
                }
              },
            ),
            IconButton(
              icon: Icon(Icons.refresh),
              onPressed: () => _loadDiagnosisRecords(),
            ),
          ],
        ),
        if (selectMode)
          TextButton(
            onPressed: () {
              setState(() {
                if (selectedIds.length == records.length) {
                  selectedIds.clear();
                } else {
                  selectedIds = records.map((r) => r.id).toSet();
                }
              });
            },
            child: Text(
              selectedIds.length == records.length ? "전체 해제" : "전체 선택",
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
          ),
      ],
    );
  }

  Widget _buildDiagnosisList() {
    if (isLoading) {
      return Center(child: CircularProgressIndicator());
    }

    if (records.isEmpty) {
      return Center(child: Text("진단 내역이 없습니다."));
    }

    return RefreshIndicator(
      onRefresh: _loadDiagnosisRecords,
      child: ListView.builder(
        itemCount: records.length,
        itemBuilder: (context, index) {
          final record = records[index];
          final isExpanded = expandedIds.contains(record.id);
          final dateStr =
              "${record.createdAt.year % 100}.${_two(record.createdAt.month)}.${_two(record.createdAt.day)} "
              "${_two(record.createdAt.hour)}:${_two(record.createdAt.minute)}";

          return Card(
            margin: EdgeInsets.only(bottom: 12),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(12),
            ),
            elevation: 2,
            child: InkWell(
              onTap: () {
                setState(() {
                  if (selectMode) {
                    selectedIds.contains(record.id)
                        ? selectedIds.remove(record.id)
                        : selectedIds.add(record.id);
                  } else {
                    isExpanded
                        ? expandedIds.remove(record.id)
                        : expandedIds.add(record.id);
                  }
                });
              },
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        if (selectMode)
                          Padding(
                            padding: const EdgeInsets.only(right: 8.0),
                            child: Checkbox(
                              value: selectedIds.contains(record.id),
                              onChanged: (checked) {
                                setState(() {
                                  checked == true
                                      ? selectedIds.add(record.id)
                                      : selectedIds.remove(record.id);
                                });
                              },
                            ),
                          ),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  Text(
                                    record.wheelPosition,
                                    style: TextStyle(
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                  SizedBox(width: 8),
                                  Text(record.type == 'air' ? '공기압' : '균열'),
                                  Spacer(),
                                  Text(
                                    record.status,
                                    style: TextStyle(
                                      color: _statusColor(record.status),
                                    ),
                                  ),
                                  SizedBox(width: 4),
                                  Icon(
                                    isExpanded
                                        ? Icons.expand_less
                                        : Icons.expand_more,
                                  ),
                                ],
                              ),
                              SizedBox(height: 6),
                              Text(
                                dateStr,
                                style: TextStyle(color: Colors.grey[600]),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                    if (isExpanded) ...[
                      SizedBox(height: 12),
                      Text(
                        "진단 점수  ${record.score}점",
                        style: TextStyle(
                          fontSize: 16,
                          color: Colors.black87,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      SizedBox(height: 8),
                      Text(record.comment),
                    ],
                  ],
                ),
              ),
            ),
          );
        },
      ),
    );
  }

  String _two(int n) => n.toString().padLeft(2, '0');

  Color _statusColor(String status) {
    switch (status) {
      case '양호':
        return Colors.green;
      case '주의':
        return Colors.amber;
      case '위험':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }
}
